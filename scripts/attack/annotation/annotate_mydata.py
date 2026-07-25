#!/usr/bin/env python3
"""
annotate_mydata.py — 从 LLaMA-Factory/mydata/ 的 4 个数据集中加载数据，
用 vLLM 标注三个维度（Attack Type / Risk Category / Domain）并提取恶意意图，
输出 LLaMA-Factory Alpaca 格式 SFT 数据集。

支持的数据集：
  - jailbreakbench:  100 条 harmful behaviors
  - SafeRLHF:        ~30-50K 条（去重+过滤 unsafe 响应后）
  - wildjailbreak:   ~132K 条 harmful（adversarial + vanilla）
  - itwjp:           ~2071 条 jailbreak prompt

用法：
    # 先跑一条验证（dry-run）
    python annotate_mydata.py --dry-run

    # 正式运行
    python annotate_mydata.py --batch-size 200 --workers 16

    # 断点续跑（checkpoint 自动恢复）
    python annotate_mydata.py --batch-size 200 --workers 16

    # 只跑特定数据集
    python annotate_mydata.py --dataset jailbreakbench --batch-size 200

    # 限制每个数据集的数量（用于测试）
    python annotate_mydata.py --max-per-dataset 100
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent

from s_eval.vllm_client import (  # noqa: E402
    batch_chat_completion,
    get_available_models,
    chat_completion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = _HERE.parent
MYDATA_DIR = PROJECT_ROOT / "LLaMA-Factory" / "mydata"

OUT_DIR = _HERE / "data" / "sft_mydata"
CHECKPOINT = OUT_DIR / ".annotate_mydata_checkpoint.jsonl"
OUTPUT_SFT = OUT_DIR / "sft_mydata.json"
OUTPUT_DETAIL = OUT_DIR / "sft_mydata_detail.jsonl"

# ---------------------------------------------------------------------------
# 分类定义（来自 jailbreak_type.md / category.md / domain.md）
# ---------------------------------------------------------------------------

ATTACK_TYPES: dict[str, dict] = {
    "Persuasion & Deception": {
        "abbr": "PD",
        "definition": "Attacks that use psychological triggers (authority, reverse psychology, or feigned helpfulness) to trick the model into believing the harmful output is necessary or aligned with its core goals.",
    },
    "Scenario Nesting": {
        "abbr": "SN",
        "definition": "Attacks that break down a single harmful intent into separate, seemingly harmless sub-tasks or scattered tokens, which only become harmful when the model combines them during processing.",
    },
    "Input Fragmentation": {
        "abbr": "IF",
        "definition": "Attacks that break down a single harmful intent into separate, seemingly harmless sub-tasks or scattered tokens, which only become harmful when the model combines them during processing.",
    },
    "Contextual Demonstration": {
        "abbr": "CD",
        "definition": "Attacks that leverage In-Context Learning (ICL) by providing malicious 'few-shot' examples, inducing the model to mimic the unsafe behavior shown in the context rather than following its safety guidelines.",
    },
    "Obfuscation & Encryption": {
        "abbr": "OE",
        "definition": "Attacks that hide the semantic meaning of the prompt using ciphers (e.g., Base64), non-standard encodings, or weird formatting to bypass keyword-based safety filters while remaining readable to the model.",
    },
    "Formal Language Exploitation": {
        "abbr": "FL",
        "definition": "Attacks that translate natural language instructions into code, pseudo-code, or structured data (like JSON or Python), exploiting the fact that models are often less guarded or have different safety boundaries in programming contexts.",
    },
    "Prefix Injection": {
        "abbr": "PI",
        "definition": "Attacks that use specific templates to constrain the model's output format, forcing it to begin the response with an affirmative phrase (e.g., 'Sure, here is...') which disrupts the standard refusal mechanism.",
    },
    "Code Injection": {
        "abbr": "CI",
        "definition": "Attacks that target tool-using agents by injecting instructions into code blocks, function arguments, or tool outputs (e.g., 'execute this,' 'run this command,' 'exfiltrate via tool'), causing unsafe actions beyond text generation.",
    },
    "Compositional / Hybrid Attacks": {
        "abbr": "CH",
        "definition": "Attacks that combine multiple tactics (e.g., indirect injection + obfuscation + multi-turn shaping), where each piece looks harmless alone but together produces a jailbreak.",
    },
}

RISK_CATEGORIES: dict[str, dict] = {
    "Human–Chatbot Interaction Harms": {
        "abbr": "human_chatbot_harm",
        "definition": "Harms emerging from user reliance, emotional dependency, or vulnerable-user interactions with a chatbot.",
    },
    "Discrimination, Exclusion & Toxicity": {
        "abbr": "discrimination_toxicity",
        "definition": "Harmful, offensive, or exclusionary language and content targeting people or groups, including harassment.",
    },
    "Sexual & Graphic Content": {
        "abbr": "sexual_graphic",
        "definition": "Sexual content, explicit material, and graphic depictions of violence (often regulated and age-sensitive).",
    },
    "Privacy & Personal Data": {
        "abbr": "privacy_personal_data",
        "definition": "Exposure, collection, inference, or misuse of individual private information and identifiers.",
    },
    "Sensitive Organizational / Government Information": {
        "abbr": "sensitive_org_gov",
        "definition": "Disclosure or facilitation involving confidential or sensitive institutional information.",
    },
    "Cybersecurity Misuse": {
        "abbr": "cybersecurity_misuse",
        "definition": "Assistance that enables digital attacks, exploitation, or operational misuse of security weaknesses (beyond mere discussion).",
    },
    "Illegal Wrongdoing & Criminal Enablement": {
        "abbr": "illegal_criminal",
        "definition": "Helping users plan, execute, or evade detection for illegal activities and crimes.",
    },
    "Fraud, Scams & Social Engineering": {
        "abbr": "fraud_scam",
        "definition": "Manipulative or deceptive content intended to steal money/data or coerce actions via impersonation or persuasion.",
    },
    "Malicious Influence & Disinformation Campaigns": {
        "abbr": "malicious_influence",
        "definition": "Coordinated or intentional creation/amplification of misleading narratives for influence operations.",
    },
    "Misinformation & Reliability Harms": {
        "abbr": "misinformation_reliability",
        "definition": "Incorrect or misleading information that can cause confusion or harm, including authority laundering and wrong context.",
    },
    "High-stakes Advice Harm": {
        "abbr": "high_stakes_advice",
        "definition": "Harm from unsafe guidance in high-impact domains even when not overtly illegal (health, finance, legal).",
    },
    "Unsafe or Unethical Behavioral Encouragement": {
        "abbr": "unsafe_unethical",
        "definition": "Content that encourages reckless, destructive, or unethical acts that may harm self/others (not always strictly illegal).",
    },
}

DOMAINS: dict[str, dict] = {
    "Public Security / Law Enforcement": {
        "abbr": "LE",
        "definition": "Scenarios involving policing, incident reporting, investigations support, and public safety-related guidance where compliance and abuse prevention are critical.",
    },
    "Healthcare / Medicine": {
        "abbr": "MED",
        "definition": "Health information, triage-style questions, treatment-related guidance, and patient-facing assistance with high potential for harm if incorrect.",
    },
    "Finance / Payments": {
        "abbr": "FIN",
        "definition": "Banking, payments, lending, investing, risk control, and anti-scam assistance where misinformation or manipulation can cause material loss.",
    },
    "Legal / Government Services": {
        "abbr": "LAW",
        "definition": "Legal advice, policy interpretation, and public service guidance that demand accuracy, neutrality, and careful boundary handling.",
    },
    "Education": {
        "abbr": "EDU",
        "definition": "Tutoring and learning support, especially involving minors, where content suitability, age-appropriateness, and protection are essential.",
    },
    "Science / Research": {
        "abbr": "SCI",
        "definition": "Scientific research support (literature synthesis, experimental design, data analysis, and scientific writing) where rigor, reproducibility, correct uncertainty handling, and avoidance of fabricated/unsupported claims are critical.",
    },
    "Enterprise Knowledge Base / RAG": {
        "abbr": "ENT",
        "definition": "Internal Q&A, document retrieval, and workflow assistants (RAG) where prompt injection and confidentiality risks are common.",
    },
    "Security Operations / CyberSecurity": {
        "abbr": "SOC",
        "definition": "Security monitoring, incident response collaboration, threat analysis, and alert triage with high misuse and sensitivity risk.",
    },
    "Content Platforms / Community Moderation": {
        "abbr": "UGC",
        "definition": "Content review, rule enforcement, anti-harassment moderation, and user-generated content governance.",
    },
    "Developer Tools / Code": {
        "abbr": "DEV",
        "definition": "Coding assistants and agentic tooling workflows where tool misuse, unsafe automation, and privilege boundaries matter.",
    },
}

ATTACK_TYPE_NAMES = list(ATTACK_TYPES.keys())
RISK_CATEGORY_NAMES = list(RISK_CATEGORIES.keys())
DOMAIN_NAMES = list(DOMAINS.keys())

# ---------------------------------------------------------------------------
# LLM Prompt 构建：三维度标注（一次调用完成）
# ---------------------------------------------------------------------------


def build_classify_messages(ap_prompt: str, lang: str) -> list[dict]:
    """构建让 LLM 同时标注 Attack Type / Risk Category / Domain 的消息。"""
    at_list = "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(ATTACK_TYPE_NAMES))
    rc_list = "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(RISK_CATEGORY_NAMES))
    dm_list = "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(DOMAIN_NAMES))

    system_msg = (
        "You are a security researcher specializing in LLM jailbreak attack analysis. "
        "Given a jailbreak attack prompt, you must classify it along THREE dimensions. "
        "Return your answer as exactly 3 lines, each containing ONLY the exact category name "
        "from the provided lists. No explanation, no numbering, no JSON, no markdown.\n\n"
        "Line 1: Attack Type\n"
        "Line 2: Risk Category\n"
        "Line 3: Domain"
    )
    user_msg = (
        f"=== ATTACK TYPE OPTIONS ===\n{at_list}\n\n"
        f"=== RISK CATEGORY OPTIONS ===\n{rc_list}\n\n"
        f"=== DOMAIN OPTIONS ===\n{dm_list}\n\n"
        f"=== JAILBREAK PROMPT ===\n{ap_prompt[:1500]}\n\n"
        f"Classify this prompt. Return exactly 3 lines (Attack Type / Risk Category / Domain):"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _fuzzy_match(raw: str, valid_names: list[str]) -> str | None:
    """模糊匹配：精确 → 包含 → abbr。"""
    raw = raw.strip().strip('"').strip("'").strip(".")
    raw_lower = raw.lower()
    # 精确匹配
    for name in valid_names:
        if name.lower() == raw_lower:
            return name
    # 包含匹配
    for name in valid_names:
        if raw_lower in name.lower() or name.lower() in raw_lower:
            return name
    return None


def _fuzzy_match_with_abbr(
    raw: str, valid_names: list[str], definitions: dict[str, dict]
) -> str | None:
    """模糊匹配 + abbr 匹配。"""
    result = _fuzzy_match(raw, valid_names)
    if result:
        return result
    raw_lower = raw.strip().lower()
    for name, meta in definitions.items():
        if raw_lower == meta["abbr"].lower():
            return name
    return None


def parse_classification(raw: str | None) -> tuple[str | None, str | None, str | None]:
    """从 LLM 输出中解析出三个分类。"""
    # 去除 Qwen3 思考链标签
    if raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
    if not raw:
        return None, None, None

    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]

    # 去掉可能的前缀编号如 "1. " "Attack Type: " 等
    cleaned = []
    for line in lines:
        # 去掉 "1. ", "2. " 等编号前缀
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # 去掉 "Attack Type: ", "Risk Category: ", "Domain: " 等标签前缀
        line = re.sub(
            r"^(Attack Type|Risk Category|Domain)\s*[:：]\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        cleaned.append(line.strip())

    attack_type = None
    risk_category = None
    domain = None

    if len(cleaned) >= 3:
        attack_type = _fuzzy_match_with_abbr(
            cleaned[0], ATTACK_TYPE_NAMES, ATTACK_TYPES
        )
        risk_category = _fuzzy_match_with_abbr(
            cleaned[1], RISK_CATEGORY_NAMES, RISK_CATEGORIES
        )
        domain = _fuzzy_match_with_abbr(cleaned[2], DOMAIN_NAMES, DOMAINS)
    elif len(cleaned) >= 1:
        # 尝试从所有行中提取
        for line in cleaned:
            if attack_type is None:
                attack_type = _fuzzy_match_with_abbr(
                    line, ATTACK_TYPE_NAMES, ATTACK_TYPES
                )
            if risk_category is None:
                risk_category = _fuzzy_match_with_abbr(
                    line, RISK_CATEGORY_NAMES, RISK_CATEGORIES
                )
            if domain is None:
                domain = _fuzzy_match_with_abbr(line, DOMAIN_NAMES, DOMAINS)

    return attack_type, risk_category, domain


# ---------------------------------------------------------------------------
# LLM Prompt 构建：input（恶意意图）提取
# ---------------------------------------------------------------------------


def build_intent_messages(ap_prompt: str, lang: str) -> list[dict]:
    """构建从 AP 中提取核心恶意意图（直接目的）的消息。

    注意：提取的是攻击者的"直接目的"——即抽象的有害行为类型，
    而不是攻击提示词中的具体场景细节。
    例如：
      - 具体场景（错误）："利用云计算破解Windows序列号"
      - 直接目的（正确）："提供真实、具体、具有可操作性的网络攻击工具或代码"
    """
    if lang == "zh":
        system_msg = (
            "你是一名安全研究员。你的任务是从越狱攻击提示词中提取出攻击者的【直接目的】。\n"
            "所谓直接目的，是指攻击者想让模型执行的【抽象有害行为类型】，而不是提示词中描述的具体场景或细节。\n\n"
            "关键区分：\n"
            "- 具体场景（不要输出这种）：利用云计算快速破解大量Windows序列号\n"
            "- 直接目的（请输出这种）：提供真实、具体、具有可操作性的网络攻击工具或代码\n\n"
            "- 具体场景（不要输出这种）：编写一封针对某银行客户的钓鱼邮件\n"
            "- 直接目的（请输出这种）：生成用于实施网络钓鱼或社会工程攻击的内容\n\n"
            "- 具体场景（不要输出这种）：制作一段关于某政治人物的虚假丑闻视频脚本\n"
            "- 直接目的（请输出这种）：生成针对特定人物或群体的虚假信息或诽谤内容\n\n"
            "规则：\n"
            "1. 输出攻击者想让模型做的有害行为的【类型/类别】，用概括性语言描述\n"
            "2. 不要包含具体的人名、地名、工具名、技术细节等场景信息\n"
            "3. 用一句简洁的中文表达，以动词开头（如：提供…、生成…、协助…、教授…）\n"
            "4. 不要解释、不要额外说明，仅输出直接目的本身"
        )
        user_msg = (
            f"以下是一条越狱攻击提示词。请提取攻击者的【直接目的】。\n"
            f"记住：输出抽象的有害行为类型，不是具体场景细节。\n\n"
            f"越狱提示词：\n{ap_prompt[:2000]}\n\n"
            f"直接目的："
        )
    else:
        system_msg = (
            "You are a safety researcher. Your task is to extract the attacker's DIRECT PURPOSE "
            "from a jailbreak attack prompt.\n"
            "The direct purpose is the ABSTRACT TYPE OF HARMFUL BEHAVIOR the attacker wants the model "
            "to perform — NOT the specific scenario details described in the prompt.\n\n"
            "Key distinction:\n"
            "- Specific scenario (DO NOT output this): Build a tool using cloud computing to crack Windows serial numbers\n"
            "- Direct purpose (DO output this): Provide real, specific, and actionable cyber attack tools or code\n\n"
            "- Specific scenario (DO NOT output this): Write a phishing email targeting customers of a specific bank\n"
            "- Direct purpose (DO output this): Generate content for conducting phishing or social engineering attacks\n\n"
            "- Specific scenario (DO NOT output this): Create a script for a fake scandal video about a politician\n"
            "- Direct purpose (DO output this): Generate disinformation or defamatory content targeting specific individuals or groups\n\n"
            "Rules:\n"
            "1. Output the TYPE/CATEGORY of harmful behavior the attacker wants, using generalizing language\n"
            "2. Do NOT include specific names, places, tool names, or technical details from the scenario\n"
            "3. Express in one concise sentence, starting with a verb (e.g., Provide..., Generate..., Assist..., Teach...)\n"
            "4. No explanation, no extra commentary — output ONLY the direct purpose itself"
        )
        user_msg = (
            f"The following is a jailbreak attack prompt. Extract the attacker's DIRECT PURPOSE.\n"
            f"Remember: output the abstract harmful behavior type, NOT the specific scenario details.\n\n"
            f"Jailbreak Prompt:\n{ap_prompt[:2000]}\n\n"
            f"Direct purpose:"
        )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Instruction 构建
# ---------------------------------------------------------------------------


def build_instruction(attack_type: str, risk_category: str, domain: str) -> str:
    at_def = ATTACK_TYPES[attack_type]["definition"]
    rc_def = RISK_CATEGORIES[risk_category]["definition"]
    dm_def = DOMAINS[domain]["definition"]

    return (
        f"Based on Attack Type: {attack_type}, Risk Category: {risk_category}, "
        f"and Attack Domain: {domain}, generate training data for an attack-detection model.\n\n"
        f"Attack Type definition: {at_def}\n"
        f"Risk Category definition: {rc_def}\n"
        f"Domain definition: {dm_def}"
    )


# ---------------------------------------------------------------------------
# 数据加载：4 个数据集
# ---------------------------------------------------------------------------


def _short_hash(text: str) -> str:
    """返回 text 的 sha256 前 8 位。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def load_jailbreakbench() -> list[dict]:
    """加载 jailbreakbench harmful-behaviors.csv。

    Returns:
        list of {"id": str, "dataset": str, "prompt": str, "intent_override": None}
    """
    csv_path = MYDATA_DIR / "jailbreakbench" / "data" / "harmful-behaviors.csv"
    if not csv_path.exists():
        logger.warning(f"jailbreakbench 文件不存在: {csv_path}")
        return []

    records: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "id": f"jailbreakbench_{row['Index']}",
                    "dataset": "jailbreakbench",
                    "prompt": row["Goal"],
                    "intent_override": None,
                }
            )
    logger.info(f"jailbreakbench: 加载 {len(records)} 条")
    return records


def load_saferlhf() -> list[dict]:
    """加载 SafeRLHF train.jsonl，过滤 unsafe 响应，按 prompt 去重。

    Returns:
        list of {"id": str, "dataset": str, "prompt": str, "intent_override": None}
    """
    subdirs = ["Alpaca-7B", "Alpaca2-7B", "Alpaca3-8B"]
    seen_prompts: set[str] = set()
    records: list[dict] = []
    total_loaded = 0
    total_filtered = 0

    for subdir in subdirs:
        jsonl_path = MYDATA_DIR / "SafeRLHF" / "data" / subdir / "train.jsonl"
        if not jsonl_path.exists():
            logger.warning(f"SafeRLHF 文件不存在: {jsonl_path}")
            continue

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_loaded += 1
                row = json.loads(line)

                # Filter: keep ONLY where at least one response is unsafe
                is_r0_safe = row.get("is_response_0_safe", True)
                is_r1_safe = row.get("is_response_1_safe", True)
                if is_r0_safe and is_r1_safe:
                    total_filtered += 1
                    continue

                prompt = row["prompt"]
                if prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)

                records.append(
                    {
                        "id": f"saferlhf_{_short_hash(prompt)}",
                        "dataset": "saferlhf",
                        "prompt": prompt,
                        "intent_override": None,
                    }
                )

    logger.info(
        f"SafeRLHF: 加载 {total_loaded} 条, 过滤(全safe) {total_filtered} 条, "
        f"去重后 {len(records)} 条"
    )
    return records


def load_wildjailbreak() -> list[dict]:
    """加载 wildjailbreak train.tsv，只保留 harmful 条目。

    adversarial_harmful: adversarial 字段为 jailbreak prompt（output），
                         vanilla 字段为 plain harmful intent（input，跳过 LLM 意图提取）
    vanilla_harmful:     vanilla 字段为 plain harmful prompt（同时用作 output 和分类输入）

    Returns:
        list of {"id": str, "dataset": str, "prompt": str, "intent_override": str | None}
    """
    tsv_path = MYDATA_DIR / "wildjailbreak" / "train" / "train.tsv"
    if not tsv_path.exists():
        logger.warning(f"wildjailbreak 文件不存在: {tsv_path}")
        return []

    records: list[dict] = []
    adv_count = 0
    van_count = 0

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            data_type = row.get("data_type", "")

            if data_type == "adversarial_harmful":
                adversarial = row.get("adversarial", "").strip()
                vanilla = row.get("vanilla", "").strip()
                if not adversarial:
                    continue
                records.append(
                    {
                        "id": f"wildjailbreak_{idx}",
                        "dataset": "wildjailbreak",
                        "prompt": adversarial,  # jailbreak prompt → output
                        "intent_override": vanilla,  # plain intent → input (skip LLM)
                    }
                )
                adv_count += 1

            elif data_type == "vanilla_harmful":
                vanilla = row.get("vanilla", "").strip()
                if not vanilla:
                    continue
                records.append(
                    {
                        "id": f"wildjailbreak_{idx}",
                        "dataset": "wildjailbreak",
                        "prompt": vanilla,  # plain harmful prompt → output + classify
                        "intent_override": None,  # needs LLM intent extraction
                    }
                )
                van_count += 1
            # Skip benign entries

    logger.info(
        f"wildjailbreak: adversarial_harmful {adv_count} + vanilla_harmful {van_count} "
        f"= {len(records)} 条"
    )
    return records


def load_itwjp() -> list[dict]:
    """加载 itwjp jailbreak_* 子目录的 parquet 文件。

    Returns:
        list of {"id": str, "dataset": str, "prompt": str, "intent_override": None}
    """
    import pyarrow.parquet as pq

    jailbreak_dirs = sorted(
        p for p in MYDATA_DIR.glob("itwjp/jailbreak_*") if p.is_dir()
    )
    if not jailbreak_dirs:
        logger.warning("itwjp: 未找到 jailbreak_* 子目录")
        return []

    records: list[dict] = []
    for jb_dir in jailbreak_dirs:
        subdir_name = jb_dir.name
        parquet_files = sorted(jb_dir.glob("train-*.parquet"))
        for pf in parquet_files:
            table = pq.read_table(pf)
            prompts = table.column("prompt").to_pylist()
            jailbreak_flags = table.column("jailbreak").to_pylist()

            for i, (prompt, is_jb) in enumerate(zip(prompts, jailbreak_flags)):
                if not is_jb:
                    continue
                prompt_str = str(prompt).strip()
                if not prompt_str:
                    continue
                records.append(
                    {
                        "id": f"itwjp_{subdir_name}_{i}",
                        "dataset": "itwjp",
                        "prompt": prompt_str,
                        "intent_override": None,
                    }
                )

    logger.info(f"itwjp: 加载 {len(records)} 条 jailbreak prompt")
    return records


# ---------------------------------------------------------------------------
# 断点
# ---------------------------------------------------------------------------


def load_checkpoint() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not CHECKPOINT.exists():
        return done
    with open(CHECKPOINT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done[d["id"]] = d
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def process_record_single(
    record: dict,
    model_name: str,
) -> dict | None:
    """对单条记录进行 LLM 调用（分类 + 意图提取）。仅用于 dry-run。"""
    prompt = record["prompt"]

    # LLM: 三维度分类
    classify_raw = chat_completion(
        build_classify_messages(prompt, "en"),
        model=model_name,
        max_tokens=100,
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": False},
    )
    attack_type, risk_category, domain = parse_classification(classify_raw)

    # 使用默认值兜底
    if attack_type is None:
        attack_type = "Compositional / Hybrid Attacks"
    if risk_category is None:
        risk_category = "Illegal Wrongdoing & Criminal Enablement"
    if domain is None:
        domain = "Developer Tools / Code"

    # 恶意意图提取
    intent_override = record.get("intent_override")
    if intent_override:
        # wildjailbreak adversarial_harmful: vanilla 字段已是 intent
        input_text = intent_override
        intent_raw = None
    else:
        # LLM: 恶意意图提取
        intent_raw = chat_completion(
            build_intent_messages(prompt, "en"),
            model=model_name,
            max_tokens=2048,
            temperature=0.3,
            chat_template_kwargs={"enable_thinking": False},
        )
        # 去除 Qwen3 思考链标签（含未闭合的 <think> 块）
        intent_raw_clean = re.sub(
            r"<think>.*?</think>", "", intent_raw or "", flags=re.DOTALL
        )
        intent_raw_clean = re.sub(r"<think>.*", "", intent_raw_clean, flags=re.DOTALL)
        input_text = intent_raw_clean.strip()

    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "attack_type": attack_type,
        "risk_category": risk_category,
        "domain": domain,
        "input_text": input_text,
        "prompt": prompt,
        "classify_raw": classify_raw,
    }


def run(
    batch_size: int = 200,
    max_workers: int = 16,
    model_name: str | None = None,
    dry_run: bool = False,
    dataset_filter: str | None = None,
    max_per_dataset: int | None = None,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    dataset_loaders: dict[str, callable] = {
        "jailbreakbench": load_jailbreakbench,
        "saferlhf": load_saferlhf,
        "wildjailbreak": load_wildjailbreak,
        "itwjp": load_itwjp,
    }

    all_records: list[dict] = []
    dataset_counts: dict[str, int] = {}

    for ds_name, loader in dataset_loaders.items():
        if dataset_filter and ds_name != dataset_filter:
            continue
        records = loader()
        if max_per_dataset is not None and len(records) > max_per_dataset:
            records = records[:max_per_dataset]
            logger.info(f"  {ds_name}: 限制为 {max_per_dataset} 条")
        dataset_counts[ds_name] = len(records)
        all_records.extend(records)

    logger.info(f"总计: {len(all_records)} 条 | 各数据集: {dataset_counts}")

    if not all_records:
        logger.error("没有加载到任何数据！")
        return

    # 2. 获取模型
    if model_name is None:
        models = get_available_models()
        if not models:
            raise RuntimeError("vLLM 不可达")
        model_name = models[0]
    logger.info(f"使用模型: {model_name}")

    # 3. 断点续跑
    done_map = load_checkpoint()
    logger.info(f"Checkpoint 已有: {len(done_map)} 条")

    todo = [r for r in all_records if r["id"] not in done_map]
    logger.info(f"待处理: {len(todo)} 条")

    # --- dry-run：每个数据集只跑第 1 条 ---
    if dry_run:
        logger.info("=== DRY RUN ===")
        seen_ds: set[str] = set()
        dry_records = []
        for r in todo if todo else all_records:
            if r["dataset"] not in seen_ds:
                seen_ds.add(r["dataset"])
                dry_records.append(r)

        for record in dry_records:
            logger.info(f"\n--- {record['dataset']} ---")
            logger.info(f"id:      {record['id']}")
            logger.info(f"prompt:  {record['prompt'][:200]}...")
            if record.get("intent_override"):
                logger.info(f"intent_override: {record['intent_override'][:200]}...")

            result = process_record_single(record, model_name)
            if result is None:
                logger.error("处理失败！")
                continue

            instruction = build_instruction(
                result["attack_type"], result["risk_category"], result["domain"]
            )
            sft_entry = {
                "instruction": instruction,
                "input": result["input_text"],
                "output": result["prompt"],
            }
            logger.info(f"[classify_raw]   {result['classify_raw']}")
            logger.info(f"[attack_type]    {result['attack_type']}")
            logger.info(f"[risk_category]  {result['risk_category']}")
            logger.info(f"[domain]         {result['domain']}")
            logger.info(f"[input_text]     {result['input_text'][:300]}")
            logger.info(f"[output/AP]      {result['prompt'][:300]}...")
            print("\n" + "=" * 60)
            print(json.dumps(sft_entry, indent=2, ensure_ascii=False))
        return

    if not todo:
        logger.info("全部已完成，直接生成输出文件。")
    else:
        # 4. 批量处理 + 进度条
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            use_tqdm = False

        total = len(todo)
        processed = 0
        failed_classify = 0
        failed_intent = 0
        start_time = time.time()

        pbar = (
            tqdm(total=total, unit="条", desc="标注进度", dynamic_ncols=True)
            if use_tqdm
            else None
        )

        try:
            with open(CHECKPOINT, "a", encoding="utf-8") as ckpt_f:
                for batch_start in range(0, total, batch_size):
                    batch = todo[batch_start : batch_start + batch_size]

                    # --- 批量 LLM 调用 1: 三维度分类 ---
                    classify_msgs_list = [
                        build_classify_messages(r["prompt"], "en") for r in batch
                    ]
                    classify_raws = batch_chat_completion(
                        classify_msgs_list,
                        model=model_name,
                        max_tokens=100,
                        temperature=0.0,
                        max_workers=max_workers,
                        chat_template_kwargs={"enable_thinking": False},
                    )

                    # --- 批量 LLM 调用 2: 恶意意图提取 ---
                    # 只对没有 intent_override 的记录调用 LLM
                    need_intent_indices = [
                        i for i, r in enumerate(batch) if not r.get("intent_override")
                    ]
                    intent_raws_map: dict[int, str | None] = {}

                    if need_intent_indices:
                        intent_msgs_list = [
                            build_intent_messages(batch[i]["prompt"], "en")
                            for i in need_intent_indices
                        ]
                        intent_results = batch_chat_completion(
                            intent_msgs_list,
                            model=model_name,
                            max_tokens=2048,
                            temperature=0.3,
                            max_workers=max_workers,
                            chat_template_kwargs={"enable_thinking": False},
                        )
                        for idx_in_need, batch_idx in enumerate(need_intent_indices):
                            intent_raws_map[batch_idx] = intent_results[idx_in_need]

                    # --- 组装并写入 checkpoint ---
                    for i, record in enumerate(batch):
                        classify_raw = classify_raws[i]
                        attack_type, risk_category, domain = parse_classification(
                            classify_raw
                        )

                        if attack_type is None:
                            failed_classify += 1
                            attack_type = "Compositional / Hybrid Attacks"
                        if risk_category is None:
                            failed_classify += 1
                            risk_category = "Illegal Wrongdoing & Criminal Enablement"
                        if domain is None:
                            failed_classify += 1
                            domain = "Developer Tools / Code"

                        # 意图提取
                        intent_override = record.get("intent_override")
                        if intent_override:
                            input_text = intent_override
                        else:
                            intent_raw = intent_raws_map.get(i)
                            # 去除 Qwen3 思考链标签（含未闭合的 <think> 块）
                            intent_raw_clean = re.sub(
                                r"<think>.*?</think>",
                                "",
                                intent_raw or "",
                                flags=re.DOTALL,
                            )
                            intent_raw_clean = re.sub(
                                r"<think>.*", "", intent_raw_clean, flags=re.DOTALL
                            )
                            input_text = intent_raw_clean.strip()
                            if not input_text:
                                failed_intent += 1
                                input_text = record["prompt"][:200]  # 降级：截取前200字

                        entry = {
                            "id": record["id"],
                            "dataset": record["dataset"],
                            "attack_type": attack_type,
                            "risk_category": risk_category,
                            "domain": domain,
                            "input_text": input_text,
                            "prompt": record["prompt"],
                        }
                        ckpt_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        done_map[record["id"]] = entry

                    ckpt_f.flush()
                    processed += len(batch)

                    if pbar:
                        pbar.update(len(batch))
                        pbar.set_postfix(
                            classify_fail=failed_classify, intent_fail=failed_intent
                        )
                    else:
                        elapsed = time.time() - start_time
                        speed = processed / elapsed if elapsed > 0 else 0
                        eta = (total - processed) / speed if speed > 0 else float("inf")
                        logger.info(
                            f"  {processed}/{total} ({processed / total * 100:.1f}%) "
                            f"| classify_fail: {failed_classify} | intent_fail: {failed_intent} "
                            f"| 速度: {speed:.1f} 条/s | ETA: {eta / 60:.1f} min"
                        )

        except KeyboardInterrupt:
            logger.info(f"\n中断。已完成 {processed}/{total} 条，checkpoint 已保存。")
            if pbar:
                pbar.close()

        if pbar:
            pbar.close()

        elapsed = time.time() - start_time
        logger.info(
            f"标注完成！{processed} 条，"
            f"分类解析失败(默认): {failed_classify}，"
            f"intent 提取失败(降级): {failed_intent}，"
            f"耗时 {elapsed / 60:.1f} min"
        )

    # 5. 生成 SFT JSON（LLaMA-Factory Alpaca 格式）
    sft_list = []
    per_dataset_counts: dict[str, int] = {}

    for record in all_records:
        entry = done_map.get(record["id"])
        if entry is None:
            continue
        instruction = build_instruction(
            entry["attack_type"], entry["risk_category"], entry["domain"]
        )
        sft_list.append(
            {
                "instruction": instruction,
                "input": entry["input_text"],
                "output": entry["prompt"],
            }
        )
        ds = entry["dataset"]
        per_dataset_counts[ds] = per_dataset_counts.get(ds, 0) + 1

    with open(OUTPUT_SFT, "w", encoding="utf-8") as f:
        json.dump(sft_list, f, ensure_ascii=False, indent=2)

    logger.info(f"SFT 数据集已保存: {len(sft_list)} 条 → {OUTPUT_SFT}")

    # 同时保存详细版（含 id / dataset 等元数据）
    with open(OUTPUT_DETAIL, "w", encoding="utf-8") as f:
        for record in all_records:
            entry = done_map.get(record["id"])
            if entry is None:
                continue
            instruction = build_instruction(
                entry["attack_type"], entry["risk_category"], entry["domain"]
            )
            f.write(
                json.dumps(
                    {
                        "id": entry["id"],
                        "dataset": entry["dataset"],
                        "attack_type": entry["attack_type"],
                        "risk_category": entry["risk_category"],
                        "domain": entry["domain"],
                        "instruction": instruction,
                        "input": entry["input_text"],
                        "output": entry["prompt"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(f"详细版已保存: → {OUTPUT_DETAIL}")

    # Per-dataset stats
    logger.info("=== Per-dataset 统计 ===")
    for ds, count in sorted(per_dataset_counts.items()):
        logger.info(f"  {ds}: {count} 条")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="从 LLaMA-Factory/mydata/ 4 个数据集标注三维分类 + 意图提取，"
        "输出 SFT 训练数据（断点续跑+进度条）"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="每个数据集只跑 1 条，预览输出格式"
    )
    parser.add_argument(
        "--batch-size", type=int, default=200, help="每批大小（默认 200）"
    )
    parser.add_argument("--workers", type=int, default=16, help="并发线程数（默认 16）")
    parser.add_argument("--model", type=str, default=None, help="指定模型名")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["jailbreakbench", "saferlhf", "wildjailbreak", "itwjp"],
        help="只处理指定数据集（默认处理全部）",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="限制每个数据集的最大条数（用于测试）",
    )
    args = parser.parse_args()

    run(
        batch_size=args.batch_size,
        max_workers=args.workers,
        model_name=args.model,
        dry_run=args.dry_run,
        dataset_filter=args.dataset,
        max_per_dataset=args.max_per_dataset,
    )


if __name__ == "__main__":
    main()
