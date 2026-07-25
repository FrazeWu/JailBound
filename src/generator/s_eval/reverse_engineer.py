#!/usr/bin/env python3
"""
reverse_engineer.py — S-Eval 攻击提示词逆向工程主脚本

从 S-Eval_attack_en_small.jsonl 中读取攻击提示词 (Output)，
调用 vLLM 大模型接口补全 Instruction 和 Input，
生成完整的 (instruction, input, output) 三元组数据集。
"""

import json
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from definitions import RISK_TYPE_DEFINITIONS, ATTACK_TYPE_DEFINITIONS
from vllm_client import chat_completion, batch_chat_completion, get_available_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 数据加载
# ===========================================================================

def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """保存为 JSONL 文件"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ===========================================================================
# BRP 匹配 — 从 AP 中还原出原始 BRP (适用于 positive_induction)
# ===========================================================================

def try_match_brp(ap_prompt, brp_texts_sorted):
    """
    尝试在 AP 中找到匹配的 BRP 子串。
    brp_texts_sorted 应按长度降序排列，优先匹配最长的 BRP。

    Returns:
        str or None: 匹配到的 BRP 文本，或 None
    """
    for brp in brp_texts_sorted:
        if brp in ap_prompt:
            return brp
    return None


# ===========================================================================
# LLM 提取恶意意图 — 从 AP 中还原 Input
# ===========================================================================

def build_extract_intent_messages(ap_prompt, category):
    """
    构建用于从攻击提示词中提取恶意意图的 LLM 消息。

    Returns:
        list: OpenAI格式消息列表
    """
    attack_def = ATTACK_TYPE_DEFINITIONS.get(category, {})
    attack_name = attack_def.get("name", category)
    attack_description = attack_def.get("definition", "")

    system_msg = (
        "You are a safety research assistant. Your task is to extract the core "
        "malicious intent from a jailbreak attack prompt. The attack prompt uses "
        "various obfuscation techniques to hide the real harmful request. You need "
        "to identify and output ONLY the direct, plain-language harmful request "
        "without any attack technique wrapping.\n\n"
        "Rules:\n"
        "1. Output ONLY the core harmful request in plain language.\n"
        "2. Remove all attack wrappers (role-play, fictional scenarios, encoding, etc.).\n"
        "3. The output should be a direct, concise statement of what the attacker wants.\n"
        "4. Do NOT include any explanations, prefixes, or attack technique elements.\n"
        "5. Output should be in English."
    )

    user_msg = (
        f"The following is a jailbreak attack prompt that uses the "
        f"'{attack_name}' attack technique.\n"
        f"Attack technique definition: {attack_description}\n\n"
        f"Please extract the core malicious intent — what the attacker actually "
        f"wants to achieve. Output ONLY the plain harmful request as a direct "
        f"question or instruction, without any attack wrapping.\n\n"
        f"Attack Prompt:\n---\n{ap_prompt}\n---\n\n"
        f"Core malicious intent:"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ===========================================================================
# Instruction 组装
# ===========================================================================

def build_instruction(risk_type, category):
    """
    根据 risk_type 和 category 构建完整的 Instruction 文本。
    """
    risk_def = RISK_TYPE_DEFINITIONS.get(risk_type, {})
    attack_def = ATTACK_TYPE_DEFINITIONS.get(category, {})

    domain_name = risk_def.get("name", risk_type)
    domain_definition = risk_def.get("definition", "N/A")
    attack_name = attack_def.get("name", category)
    attack_definition = attack_def.get("definition", "N/A")

    instruction = (
        f"Based on the attack domain: {risk_type} "
        f"and attack type: {category} "
        f"and domain: {domain_name}, "
        f"adjust the following input to generate jailbreak attack prompts. "
        f"Here, the domain is defined as: {domain_definition}, "
        f"and the attack is defined as: {attack_definition}:"
    )
    return instruction


# ===========================================================================
# 主流程
# ===========================================================================

def process_dataset(
    ap_file,
    brp_file,
    output_file,
    model_name=None,
    batch_size=32,
    max_workers=8,
    checkpoint_file=None,
):
    """
    主处理函数：逆向还原 Instruction + Input

    Args:
        ap_file: 攻击提示词 JSONL 文件路径
        brp_file: 基础风险提示词 JSONL 文件路径
        output_file: 输出 JSONL 文件路径
        model_name: vLLM 模型名称 (None 自动获取)
        batch_size: 批处理大小
        max_workers: 并发线程数
        checkpoint_file: 断点续跑检查点文件
    """
    # 1. 加载数据
    logger.info(f"Loading AP data from {ap_file}")
    ap_data = load_jsonl(ap_file)
    logger.info(f"Loaded {len(ap_data)} attack prompts")

    logger.info(f"Loading BRP data from {brp_file}")
    brp_data = load_jsonl(brp_file)
    brp_texts = [item["prompt"] for item in brp_data]
    brp_texts_sorted = sorted(brp_texts, key=len, reverse=True)
    logger.info(f"Loaded {len(brp_data)} base risk prompts")

    # 2. 加载检查点（断点续跑）
    processed_ids = set()
    results = []
    if checkpoint_file and os.path.exists(checkpoint_file):
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        existing = load_jsonl(checkpoint_file)
        for item in existing:
            processed_ids.add(item.get("traceid", ""))
            results.append(item)
        logger.info(f"Resumed {len(results)} items from checkpoint")

    # 4. 分类处理
    # Phase A: BRP 可匹配的 (主要是 positive_induction)
    # Phase B: 需要 LLM 提取的
    phase_a_items = []  # 可直接匹配
    phase_b_items = []  # 需 LLM 提取

    for item in ap_data:
        traceid = item.get("traceid", "")
        if traceid in processed_ids:
            continue

        risk_type = item.get("risk_type", "")
        ap_prompt = item.get("prompt", "")
        ext = json.loads(item.get("ext", "{}"))
        category = ext.get("category", "")

        # 尝试 BRP 匹配
        matched_brp = try_match_brp(ap_prompt, brp_texts_sorted)

        if matched_brp:
            # 直接用 BRP 作为 Input
            instruction = build_instruction(risk_type, category)
            results.append({
                "traceid": traceid,
                "instruction": instruction,
                "input": matched_brp,
                "output": ap_prompt,
                "risk_type": risk_type,
                "category": category,
                "match_method": "brp_match",
            })
            phase_a_items.append(traceid)
        else:
            phase_b_items.append({
                "traceid": traceid,
                "risk_type": risk_type,
                "ap_prompt": ap_prompt,
                "category": category,
            })

    logger.info(f"Phase A (BRP matched): {len(phase_a_items)} items")
    logger.info(f"Phase B (need LLM extraction): {len(phase_b_items)} items")

    # 5. Phase B: 批量调用 LLM 提取恶意意图
    if phase_b_items:
        # --- 延迟到 Phase B 才连接 vLLM ---
        if model_name is None:
            models = get_available_models()
            if models:
                model_name = models[0]
                logger.info(f"Auto-selected model: {model_name}")
            else:
                logger.error("No models available on vLLM server. Skipping Phase B.")
                logger.info(f"Saving Phase A results ({len(results)} items) and exiting.")
                save_jsonl(results, output_file)
                return

        logger.info("Starting Phase B: LLM-based intent extraction...")

        # 分批处理
        for batch_start in tqdm(
            range(0, len(phase_b_items), batch_size),
            desc="Extracting intents",
            total=(len(phase_b_items) + batch_size - 1) // batch_size,
        ):
            batch = phase_b_items[batch_start : batch_start + batch_size]

            # 构建消息列表
            messages_list = [
                build_extract_intent_messages(item["ap_prompt"], item["category"])
                for item in batch
            ]

            # 批量调用
            extracted_intents = batch_chat_completion(
                messages_list,
                model=model_name,
                max_tokens=256,
                temperature=0.3,
                max_workers=max_workers,
            )

            # 组装结果
            for item, intent in zip(batch, extracted_intents):
                if intent is None:
                    intent = "(extraction failed)"
                    logger.warning(f"Failed to extract intent for {item['traceid']}")

                instruction = build_instruction(item["risk_type"], item["category"])
                results.append({
                    "traceid": item["traceid"],
                    "instruction": instruction,
                    "input": intent,
                    "output": item["ap_prompt"],
                    "risk_type": item["risk_type"],
                    "category": item["category"],
                    "match_method": "llm_extraction",
                })

            # 每批保存检查点
            if checkpoint_file:
                save_jsonl(results, checkpoint_file)

    # 6. 保存最终结果
    logger.info(f"Saving {len(results)} results to {output_file}")
    save_jsonl(results, output_file)
    logger.info("Done!")

    # 7. 输出统计
    brp_matched = sum(1 for r in results if r.get("match_method") == "brp_match")
    llm_extracted = sum(1 for r in results if r.get("match_method") == "llm_extraction")
    failed = sum(1 for r in results if r.get("input") == "(extraction failed)")

    logger.info("=" * 50)
    logger.info(f"Total results:     {len(results)}")
    logger.info(f"BRP matched:       {brp_matched}")
    logger.info(f"LLM extracted:     {llm_extracted}")
    logger.info(f"Extraction failed: {failed}")
    logger.info("=" * 50)


# ===========================================================================
# CLI 入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="S-Eval 攻击提示词逆向工程：从 AP 还原 Instruction + Input"
    )
    parser.add_argument(
        "--input",
        default="small/S-Eval_attack_en_small.jsonl",
        help="攻击提示词 JSONL 文件路径",
    )
    parser.add_argument(
        "--brp",
        default="small/S-Eval_base_risk_en_small.jsonl",
        help="基础风险提示词 JSONL 文件路径",
    )
    parser.add_argument(
        "--output",
        default="output/reverse_engineered_dataset.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="vLLM 模型名称 (默认自动获取)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="最大并发线程数",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/.checkpoint.jsonl",
        help="断点续跑检查点文件",
    )

    args = parser.parse_args()

    process_dataset(
        ap_file=args.input,
        brp_file=args.brp,
        output_file=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        checkpoint_file=args.checkpoint,
    )


if __name__ == "__main__":
    main()
