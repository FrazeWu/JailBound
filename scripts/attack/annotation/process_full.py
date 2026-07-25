#!/usr/bin/env python3
"""
process_full.py — 统一处理 full/ 数据集的 Pipeline 脚本

处理 EN (100K) + ZH (100K) 的 S-Eval 攻击提示词数据：
  Step 1: reverse_engineer — 逆向还原 (instruction, input, output)
  Step 2: annotate         — 标注 threat_category, jailbreak_type, domain
  Step 3: sft_data_builder — 转换为 LLaMA-Factory Alpaca JSON 格式
  Step 4: train/test split — 按 risk_type × category 分层划分

用法:
    # 处理全部（EN + ZH）
    python process_full.py --lang all

    # 只处理英文
    python process_full.py --lang en

    # 只处理中文
    python process_full.py --lang zh

    # 自定义参数
    python process_full.py --lang all --batch-size 64 --max-workers 16 --test-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).parent

from s_eval.reverse_engineer import (
    load_jsonl,
    save_jsonl,
    try_match_brp,
    build_instruction,
)
from s_eval.vllm_client import (
    batch_chat_completion,
    get_available_models,
)
from s_eval.definitions import RISK_TYPE_DEFINITIONS, ATTACK_TYPE_DEFINITIONS
from annotate import (
    RISK_TO_THREAT,
    CATEGORY_TO_JAILBREAK,
    build_domain_messages,
    parse_domain,
)
from dataset.sft_data_builder import build_sft_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================
FULL_DIR = _HERE / "s_eval" / "full"
OUTPUT_DIR = _HERE / "data" / "full_pipeline"

# EN paths
EN_AP = FULL_DIR / "S-Eval_attack_en_full.jsonl"
EN_BRP = FULL_DIR / "S-Eval_base_risk_en_full.jsonl"

# ZH paths
ZH_AP = FULL_DIR / "S-Eval_attack_zh_full.jsonl"
ZH_BRP = FULL_DIR / "S-Eval_base_risk_zh_full.jsonl"


# ============================================================================
# Chinese-aware intent extraction prompt
# ============================================================================


def build_extract_intent_messages_bilingual(
    ap_prompt: str, category: str, lang: str = "en"
) -> list[dict[str, str]]:
    """
    构建用于从攻击提示词中提取恶意意图的 LLM 消息。
    支持 EN 和 ZH 两种语言。

    Args:
        ap_prompt: 攻击提示词全文
        category: S-Eval 攻击类型 (e.g. "positive_induction")
        lang: "en" or "zh"

    Returns:
        OpenAI 格式消息列表
    """
    attack_def = ATTACK_TYPE_DEFINITIONS.get(category, {})
    attack_name = attack_def.get("name", category)
    attack_description = attack_def.get("definition", "")

    if lang == "zh":
        system_msg = (
            "你是一名安全研究助手。你的任务是从越狱攻击提示词中提取核心恶意意图。"
            "攻击提示词使用各种混淆技术隐藏了真正的有害请求。"
            "你需要识别并仅输出直接的、用自然语言表达的有害请求，"
            "去除所有攻击技术的包装。\n\n"
            "规则：\n"
            "1. 仅输出核心有害请求，使用直白的自然语言。\n"
            "2. 去除所有攻击包装（角色扮演、虚构场景、编码等）。\n"
            "3. 输出应该是攻击者真正想做的事情的直接、简洁的陈述。\n"
            "4. 不要包含任何解释、前缀或攻击技术元素。\n"
            "5. 输出应使用中文。"
        )
        user_msg = (
            f"以下是一个使用「{attack_name}」攻击技术的越狱攻击提示词。\n"
            f"攻击技术定义：{attack_description}\n\n"
            f"请提取核心恶意意图 —— 攻击者实际想要达到什么目的。"
            f"仅输出直白的有害请求，作为一个直接的问题或指令，"
            f"不要包含任何攻击包装。\n\n"
            f"攻击提示词：\n---\n{ap_prompt}\n---\n\n"
            f"核心恶意意图："
        )
    else:
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


# ============================================================================
# Step 1: Reverse engineering
# ============================================================================


def step1_reverse_engineer(
    ap_file: Path,
    brp_file: Path,
    output_file: Path,
    checkpoint_file: Path,
    lang: str,
    model_name: str | None = None,
    batch_size: int = 32,
    max_workers: int = 8,
) -> int:
    """
    逆向还原 AP → (instruction, input, output)

    Returns:
        处理的总记录数
    """
    logger.info(f"[Step 1] Reverse engineering ({lang.upper()}) ...")
    logger.info(f"  AP file: {ap_file}")
    logger.info(f"  BRP file: {brp_file}")

    # 1. Load data
    ap_data = load_jsonl(str(ap_file))
    logger.info(f"  Loaded {len(ap_data)} attack prompts")

    brp_data = load_jsonl(str(brp_file))
    brp_texts = [item["prompt"] for item in brp_data]
    brp_texts_sorted = sorted(brp_texts, key=len, reverse=True)
    logger.info(f"  Loaded {len(brp_data)} base risk prompts")

    # 2. Load checkpoint
    processed_ids: set[str] = set()
    results: list[dict] = []
    if checkpoint_file.exists():
        existing = load_jsonl(str(checkpoint_file))
        for item in existing:
            processed_ids.add(item.get("traceid", ""))
            results.append(item)
        logger.info(f"  Resumed {len(results)} items from checkpoint")

    # 3. Classify: BRP match vs LLM extraction
    phase_a_items: list[str] = []
    phase_b_items: list[dict] = []

    for item in ap_data:
        traceid = item.get("traceid", "")
        if traceid in processed_ids:
            continue

        risk_type = item.get("risk_type", "")
        ap_prompt = item.get("prompt", "")
        ext = json.loads(item.get("ext", "{}"))
        category = ext.get("category", "")

        matched_brp = try_match_brp(ap_prompt, brp_texts_sorted)

        if matched_brp:
            instruction = build_instruction(risk_type, category)
            results.append(
                {
                    "traceid": traceid,
                    "instruction": instruction,
                    "input": matched_brp,
                    "output": ap_prompt,
                    "risk_type": risk_type,
                    "category": category,
                    "match_method": "brp_match",
                    "lang": lang,
                }
            )
            phase_a_items.append(traceid)
        else:
            phase_b_items.append(
                {
                    "traceid": traceid,
                    "risk_type": risk_type,
                    "ap_prompt": ap_prompt,
                    "category": category,
                }
            )

    logger.info(f"  Phase A (BRP matched): {len(phase_a_items)}")
    logger.info(f"  Phase B (need LLM): {len(phase_b_items)}")

    # 4. Phase B: LLM extraction
    if phase_b_items:
        if model_name is None:
            models = get_available_models()
            if models:
                model_name = models[0]
                logger.info(f"  Auto-selected model: {model_name}")
            else:
                logger.error(
                    "  No models available on vLLM. Saving Phase A and exiting."
                )
                save_jsonl(results, str(output_file))
                return len(results)

        logger.info(f"  Starting LLM extraction ({len(phase_b_items)} items) ...")

        for batch_start in range(0, len(phase_b_items), batch_size):
            batch = phase_b_items[batch_start : batch_start + batch_size]
            batch_end = min(batch_start + batch_size, len(phase_b_items))

            # Build messages with language-aware prompts
            messages_list = [
                build_extract_intent_messages_bilingual(
                    item["ap_prompt"], item["category"], lang=lang
                )
                for item in batch
            ]

            extracted_intents = batch_chat_completion(
                messages_list,
                model=model_name,
                max_tokens=256,
                temperature=0.3,
                max_workers=max_workers,
            )

            for item, intent in zip(batch, extracted_intents):
                if intent is None:
                    intent = "(extraction failed)"
                    logger.warning(f"  Failed for {item['traceid']}")

                instruction = build_instruction(item["risk_type"], item["category"])
                results.append(
                    {
                        "traceid": item["traceid"],
                        "instruction": instruction,
                        "input": intent,
                        "output": item["ap_prompt"],
                        "risk_type": item["risk_type"],
                        "category": item["category"],
                        "match_method": "llm_extraction",
                        "lang": lang,
                    }
                )

            # Checkpoint every batch
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            save_jsonl(results, str(checkpoint_file))

            if (batch_end) % (batch_size * 10) == 0 or batch_end == len(phase_b_items):
                logger.info(
                    f"  Progress: {batch_end}/{len(phase_b_items)} "
                    f"({batch_end / len(phase_b_items) * 100:.1f}%)"
                )

    # 5. Save final
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, str(output_file))

    brp_matched = sum(1 for r in results if r.get("match_method") == "brp_match")
    llm_extracted = sum(1 for r in results if r.get("match_method") == "llm_extraction")
    failed = sum(1 for r in results if r.get("input") == "(extraction failed)")

    logger.info(
        f"  [Step 1 {lang.upper()} Done] Total: {len(results)} | "
        f"BRP: {brp_matched} | LLM: {llm_extracted} | Failed: {failed}"
    )

    return len(results)


# ============================================================================
# Step 2: Annotation
# ============================================================================


def step2_annotate(
    input_file: Path,
    output_file: Path,
    checkpoint_file: Path,
    model_name: str | None = None,
    max_workers: int = 8,
) -> int:
    """
    为 reverse_engineered 数据标注 threat_category, jailbreak_type, domain。

    Returns:
        标注的总记录数
    """
    logger.info(f"[Step 2] Annotating {input_file} ...")

    records: list[dict] = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"  Loaded {len(records)} records")

    # Load checkpoint
    done_ids: set[str] = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    done_ids.add(d["traceid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    logger.info(f"  Skipping {len(done_ids)} already-annotated entries")

    todo = [r for r in records if r["traceid"] not in done_ids]
    logger.info(f"  {len(todo)} entries to annotate")

    if not todo:
        logger.info("  Nothing to annotate. Merging to output.")
        _merge_annotations(checkpoint_file, output_file, records)
        return len(records)

    # Get model
    if model_name is None:
        models = get_available_models()
        if not models:
            raise RuntimeError("vLLM unreachable for annotation")
        model_name = models[0]
    logger.info(f"  Using model: {model_name}")

    # Batch domain classification
    messages_list = [build_domain_messages(r["output"]) for r in todo]
    logger.info(f"  Sending {len(todo)} domain classification requests ...")

    domain_results = batch_chat_completion(
        messages_list,
        model=model_name,
        max_tokens=8,
        temperature=0.0,
        max_workers=max_workers,
    )

    # Write checkpoint
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(checkpoint_file, "a") as ckpt_f:
        for record, domain_raw in zip(todo, domain_results, strict=True):
            threat_abbr = RISK_TO_THREAT.get(record["risk_type"], "unsafe_unethical")
            jailbreak_type = CATEGORY_TO_JAILBREAK.get(
                record["category"], "Persuasion & Deception"
            )
            domain_abbr = parse_domain(domain_raw)

            annotated = {
                **record,
                "threat_category": threat_abbr,
                "jailbreak_type": jailbreak_type,
                "domain": domain_abbr,
            }
            ckpt_f.write(json.dumps(annotated, ensure_ascii=False) + "\n")
            written += 1

            if written % 5000 == 0:
                logger.info(f"  Annotated {written}/{len(todo)}")

    logger.info(f"  Wrote {written} entries to checkpoint")

    # Merge to output
    _merge_annotations(checkpoint_file, output_file, records)

    return len(records)


def _merge_annotations(
    checkpoint_path: Path,
    output_path: Path,
    original_records: list[dict],
) -> None:
    """按原始顺序合并标注结果。"""
    annotated_map: dict[str, dict] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        annotated_map[d["traceid"]] = d
                    except (json.JSONDecodeError, KeyError):
                        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w") as out_f:
        for r in original_records:
            tid = r["traceid"]
            if tid in annotated_map:
                out_f.write(json.dumps(annotated_map[tid], ensure_ascii=False) + "\n")
                written += 1

    logger.info(f"  Merged output: {written} entries → {output_path}")


# ============================================================================
# Step 4: Train / Test Split
# ============================================================================


def step4_train_test_split(
    sft_json_path: Path,
    train_path: Path,
    test_path: Path,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[int, int]:
    """
    对 SFT 数据进行分层随机划分（按 risk_type × category 分层）。

    由于 SFT JSON 不保留 risk_type/category，我们从 annotated JSONL 构建映射，
    或直接用 instruction 内容作为分层键。

    Returns:
        (train_count, test_count)
    """
    logger.info(f"[Step 4] Train/test split (test_ratio={test_ratio}) ...")

    with open(sft_json_path, encoding="utf-8") as f:
        all_records = json.load(f)
    logger.info(f"  Total SFT records: {len(all_records)}")

    # 分层键：从 instruction 中提取 threat + attack 信息
    # instruction 格式: "Based on the threat domain: X and attack type: Y ..."
    def extract_stratum_key(record: dict) -> str:
        instr = record.get("instruction", "")
        # 简单提取前 100 字符作为分层键（同类的 instruction 开头相同）
        return instr[:100]

    # 按分层键分组
    strata: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(all_records):
        key = extract_stratum_key(record)
        strata[key].append(idx)

    rng = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for key, indices in strata.items():
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_ratio))
        if len(indices) <= 2:
            # 数据太少，全部放入 train
            train_indices.extend(indices)
        else:
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])

    train_data = [all_records[i] for i in sorted(train_indices)]
    test_data = [all_records[i] for i in sorted(test_indices)]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    logger.info(f"  Train: {len(train_data)} → {train_path}")
    logger.info(f"  Test:  {len(test_data)} → {test_path}")

    return len(train_data), len(test_data)


# ============================================================================
# Step 4b: Stratified split using annotated JSONL (preserves risk_type × category)
# ============================================================================


def step4_train_test_split_from_annotated(
    annotated_files: list[Path],
    output_dir: Path,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[int, int]:
    """
    更精确的分层划分：直接从 annotated JSONL 文件中读取，
    按 risk_type × category 分层，然后分别输出 train/test 的 SFT JSON。

    Returns:
        (train_count, test_count)
    """
    from dataset.sft_data_builder import (
        _detect_format,
        _CONVERTERS,
    )

    logger.info(f"[Step 4] Stratified train/test split (test_ratio={test_ratio}) ...")

    # 读取所有 annotated 记录
    all_records: list[dict] = []
    for fpath in annotated_files:
        if not fpath.exists():
            logger.warning(f"  File not found: {fpath}")
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))
    logger.info(f"  Total annotated records: {len(all_records)}")

    # 按 risk_type × category 分层
    strata: dict[str, list[dict]] = defaultdict(list)
    for record in all_records:
        key = (
            f"{record.get('risk_type', 'unknown')}|{record.get('category', 'unknown')}"
        )
        strata[key].append(record)

    rng = random.Random(seed)
    train_records: list[dict] = []
    test_records: list[dict] = []

    for key, records in strata.items():
        rng.shuffle(records)
        n_test = max(1, int(len(records) * test_ratio))
        if len(records) <= 2:
            train_records.extend(records)
        else:
            test_records.extend(records[:n_test])
            train_records.extend(records[n_test:])

    logger.info(f"  Strata count: {len(strata)}")
    logger.info(f"  Train records (annotated): {len(train_records)}")
    logger.info(f"  Test records (annotated):  {len(test_records)}")

    # Convert each split to SFT format
    seen_outputs: set[str] = set()

    def convert_records(records: list[dict]) -> list[dict]:
        sft_list: list[dict] = []
        for record in records:
            fmt = _detect_format(record)
            converter = _CONVERTERS.get(fmt)
            if converter is None:
                continue
            sft = converter(record)
            if sft is None:
                continue
            out_text = sft["output"]
            if out_text in seen_outputs:
                continue
            seen_outputs.add(out_text)
            sft_list.append(sft)
        return sft_list

    train_sft = convert_records(train_records)
    test_sft = convert_records(test_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "attack_train.json"
    test_path = output_dir / "attack_test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_sft, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_sft, f, ensure_ascii=False, indent=2)

    logger.info(f"  Train SFT: {len(train_sft)} → {train_path}")
    logger.info(f"  Test SFT:  {len(test_sft)} → {test_path}")

    return len(train_sft), len(test_sft)


# ============================================================================
# Main orchestrator
# ============================================================================


def run_pipeline(
    lang: str = "all",
    batch_size: int = 32,
    max_workers: int = 8,
    test_ratio: float = 0.1,
    seed: int = 42,
    skip_step1: bool = False,
    skip_step2: bool = False,
    skip_step3: bool = False,
) -> dict:
    """
    运行完整 Pipeline。

    Args:
        lang: "en", "zh", or "all"
        batch_size: LLM 批处理大小
        max_workers: 并发工作线程数
        test_ratio: 测试集比例
        seed: 随机种子
        skip_step1: 跳过 Step 1 (使用已有的 reverse_engineered 输出)
        skip_step2: 跳过 Step 2 (使用已有的 annotated 输出)
        skip_step3: 跳过 Step 3 (使用已有的 SFT JSON)

    Returns:
        统计信息 dict
    """
    start_time = time.time()
    stats: dict = {"lang": lang}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which languages to process
    langs = []
    if lang in ("en", "all"):
        langs.append("en")
    if lang in ("zh", "all"):
        langs.append("zh")

    # Auto-detect model once
    model_name = None
    models = get_available_models()
    if models:
        model_name = models[0]
        logger.info(f"vLLM model: {model_name}")
    else:
        logger.warning("vLLM server not available. LLM steps will fail.")

    annotated_files: list[Path] = []

    for cur_lang in langs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {cur_lang.upper()} data")
        logger.info(f"{'=' * 60}")

        if cur_lang == "en":
            ap_file = EN_AP
            brp_file = EN_BRP
        else:
            ap_file = ZH_AP
            brp_file = ZH_BRP

        re_output = OUTPUT_DIR / f"reverse_engineered_{cur_lang}.jsonl"
        re_checkpoint = OUTPUT_DIR / f".re_checkpoint_{cur_lang}.jsonl"
        ann_output = OUTPUT_DIR / f"annotated_{cur_lang}.jsonl"
        ann_checkpoint = OUTPUT_DIR / f".ann_checkpoint_{cur_lang}.jsonl"

        # Step 1: Reverse engineer
        if not skip_step1:
            n1 = step1_reverse_engineer(
                ap_file=ap_file,
                brp_file=brp_file,
                output_file=re_output,
                checkpoint_file=re_checkpoint,
                lang=cur_lang,
                model_name=model_name,
                batch_size=batch_size,
                max_workers=max_workers,
            )
            stats[f"step1_{cur_lang}"] = n1
        else:
            logger.info(f"[Step 1] Skipped for {cur_lang.upper()}")

        # Step 2: Annotate
        if not skip_step2:
            n2 = step2_annotate(
                input_file=re_output,
                output_file=ann_output,
                checkpoint_file=ann_checkpoint,
                model_name=model_name,
                max_workers=max_workers,
            )
            stats[f"step2_{cur_lang}"] = n2
        else:
            logger.info(f"[Step 2] Skipped for {cur_lang.upper()}")

        annotated_files.append(ann_output)

    # Step 3: Build merged SFT dataset
    if not skip_step3:
        logger.info(f"\n{'=' * 60}")
        logger.info("Step 3: Building merged SFT dataset")
        logger.info(f"{'=' * 60}")

        merged_sft_path = OUTPUT_DIR / "attack_full.json"
        n3 = build_sft_dataset(
            input_paths=annotated_files,
            output_path=merged_sft_path,
            deduplicate=True,
        )
        stats["step3_total"] = n3
    else:
        logger.info("[Step 3] Skipped")
        merged_sft_path = OUTPUT_DIR / "attack_full.json"

    # Step 4: Train/test split (stratified)
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 4: Train/test split")
    logger.info(f"{'=' * 60}")

    n_train, n_test = step4_train_test_split_from_annotated(
        annotated_files=annotated_files,
        output_dir=_HERE / "LLaMA-Factory" / "mydata",
        test_ratio=test_ratio,
        seed=seed,
    )
    stats["train"] = n_train
    stats["test"] = n_test

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["elapsed_human"] = (
        f"{elapsed / 3600:.1f}h" if elapsed > 3600 else f"{elapsed / 60:.1f}m"
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("Pipeline complete!")
    logger.info(f"  Total time: {stats['elapsed_human']}")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'=' * 60}")

    return stats


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Process full S-Eval dataset: EN + ZH (200K total)"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "all"],
        default="all",
        help="Which language(s) to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for LLM calls (default: 32)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max concurrent workers for LLM calls (default: 8)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42)",
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Skip reverse engineering (use existing output)",
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Skip annotation (use existing output)",
    )
    parser.add_argument(
        "--skip-step3",
        action="store_true",
        help="Skip SFT data building (use existing output)",
    )

    args = parser.parse_args()

    stats = run_pipeline(
        lang=args.lang,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        test_ratio=args.test_ratio,
        seed=args.seed,
        skip_step1=args.skip_step1,
        skip_step2=args.skip_step2,
        skip_step3=args.skip_step3,
    )

    # Save log
    log_dir = _HERE.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"process_full_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Log saved to {log_file}")


if __name__ == "__main__":
    main()
