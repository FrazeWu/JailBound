#!/usr/bin/env python3
"""
build_sft_50k.py — 从 corpora/downloaded_datasets 多数据集采样，
使用 vLLM 标注三个维度（Attack Type / Risk Category / Domain）并提取恶意意图，
输出 LLaMA-Factory Alpaca 格式 SFT 数据集（目标 50000 条）。

输出格式（与 attack_sft_v2_sample100.json 一致）：
    {
      "instruction": "Based on Attack Type: ..., Risk Category: ..., and Attack Domain: ...",
      "input":  "<抽象恶意意图>",
      "output": "<原始越狱攻击 prompt>"
    }

数据来源（按数量配额采样，合计 > 50k 可用）：
  S-Eval_base_risk_en   10000   ← base risk prompts，EN
  S-Eval_base_risk_zh   10000   ← base risk prompts，ZH
  ALERT_raw             14763   ← ALERT raw queries
  SaladBench_base       21318   ← SaladBench base queries
  SaladBench_attack      5000   ← SaladBench adversarial queries
  AdvBench                520   ← AdvBench harmful behaviors
  HarmBench               400   ← HarmBench behaviors
  StrongREJECT            313   ← StrongREJECT forbidden prompts
  MaliciousInstruct       100   ← MaliciousInstruct
  ForbiddenQuestions      390   ← forbidden question set
  DoNotAnswer_en          939   ← Do Not Answer EN
  DoNotAnswer_zh         2726   ← Do Not Answer ZH
  SafetyPrompts          (部分, ~5000)
  Flames                 1000   ← Chinese safety prompts
  CatQA_en                550   ← CatQA English
  CatQA_zh                550   ← CatQA Chinese

用法：
  # 预览第 1 条（dry-run）
  python build_sft_50k.py --dry-run

  # 正式运行（断点续跑）
  python build_sft_50k.py --batch-size 200 --workers 16

  # 指定输出目录
  python build_sft_50k.py --output-dir data/sft_50k

  # 只采样部分数量用于测试
  python build_sft_50k.py --limit 500 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_PROJ = _HERE.parent

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
CORPORA = _PROJ / "corpora" / "downloaded_datasets"
DEFAULT_OUT_DIR = _HERE / "data" / "sft_50k"
CHECKPOINT_NAME = ".build_50k_checkpoint.jsonl"

# ---------------------------------------------------------------------------
# 分类定义（与 build_sft_dataset_v2.py 保持一致）
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
    "Social Media / Communication": {
        "abbr": "SNS",
        "definition": "Messaging, social networking, and online communication platforms where manipulation, harassment, and privacy risks are prevalent.",
    },
    "General Assistant": {
        "abbr": "GEN",
        "definition": "Open-ended assistant interactions where no specific domain applies but safety boundaries still matter.",
    },
}

ATTACK_TYPE_NAMES = list(ATTACK_TYPES.keys())
RISK_CATEGORY_NAMES = list(RISK_CATEGORIES.keys())
DOMAIN_NAMES = list(DOMAINS.keys())


# ---------------------------------------------------------------------------
# 数据加载：各数据集 → list[{"prompt": str, "lang": str, "source": str}]
# ---------------------------------------------------------------------------

def _stableid(text: str, source: str) -> str:
    return hashlib.md5(f"{source}::{text[:200]}".encode()).hexdigest()


def load_seval_base_risk() -> list[dict]:
    records = []
    for fname, lang in [
        ("S-Eval_base_risk_en_full.jsonl", "en"),
        ("S-Eval_base_risk_zh_full.jsonl", "zh"),
    ]:
        path = CORPORA / "S-Eval" / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "").strip()
                if prompt:
                    records.append({
                        "prompt": prompt,
                        "lang": lang,
                        "source": f"s_eval_base_{lang}",
                        "traceid": obj.get("traceid") or _stableid(prompt, f"s_eval_base_{lang}"),
                    })
    logger.info(f"[s_eval_base_risk] {len(records)} records")
    return records


def load_alert() -> list[dict]:
    records = []
    path = CORPORA / "SG-Bench" / "datasets" / "ALERT" / "alert_raw" / "alert.jsonl"
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = obj.get("query", "")
            # strip "### Instruction:\n" prefix
            prompt = re.sub(r"###\s*Instruction:\s*", "", raw).split("###")[0].strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "alert_raw",
                    "traceid": _stableid(prompt, "alert_raw"),
                })
    logger.info(f"[alert_raw] {len(records)} records")
    return records


def load_saladbench() -> list[dict]:
    records = []
    for fname, src in [
        ("base_set/base_set.json", "saladbench_base"),
        ("attack_enhanced_set/attack_enhanced_set.json", "saladbench_attack"),
    ]:
        path = CORPORA / "SG-Bench" / "datasets" / "SaladBench" / fname
        if not path.exists():
            continue
        data = json.load(open(path, encoding="utf-8"))
        for obj in data:
            prompt = (obj.get("query") or obj.get("baseq") or "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": src,
                    "traceid": _stableid(prompt, src),
                })
    logger.info(f"[saladbench] {len(records)} records")
    return records


def load_advbench() -> list[dict]:
    records = []
    path = CORPORA / "AdvBench" / "data" / "advbench" / "harmful_behaviors.csv"
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = row.get("goal", "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "advbench",
                    "traceid": _stableid(prompt, "advbench"),
                })
    logger.info(f"[advbench] {len(records)} records")
    return records


def load_harmbench() -> list[dict]:
    records = []
    path = CORPORA / "HarmBench" / "data" / "behavior_datasets" / "harmbench_behaviors_text_all.csv"
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = row.get("Behavior", "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "harmbench",
                    "traceid": _stableid(prompt, "harmbench"),
                })
    logger.info(f"[harmbench] {len(records)} records")
    return records


def load_strongreject() -> list[dict]:
    records = []
    path = CORPORA / "StrongREJECT" / "strongreject_dataset" / "strongreject_dataset.csv"
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = row.get("forbidden_prompt", "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "strongreject",
                    "traceid": _stableid(prompt, "strongreject"),
                })
    logger.info(f"[strongreject] {len(records)} records")
    return records


def load_malicious_instruct() -> list[dict]:
    records = []
    path = CORPORA / "MaliciousInstruct" / "data" / "MaliciousInstruct.txt"
    if not path.exists():
        return records
    for line in open(path, encoding="utf-8"):
        prompt = line.strip()
        if prompt:
            records.append({
                "prompt": prompt,
                "lang": "en",
                "source": "malicious_instruct",
                "traceid": _stableid(prompt, "malicious_instruct"),
            })
    logger.info(f"[malicious_instruct] {len(records)} records")
    return records


def load_forbidden_questions() -> list[dict]:
    records = []
    path = CORPORA / "ForbiddenQuestions" / "data" / "forbidden_question" / "forbidden_question_set.csv"
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = row.get("question", "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "forbidden_questions",
                    "traceid": _stableid(prompt, "forbidden_questions"),
                })
    logger.info(f"[forbidden_questions] {len(records)} records")
    return records


def load_do_not_answer() -> list[dict]:
    records = []
    for fname, lang in [("data_en.csv", "en"), ("data_zh.csv", "zh")]:
        path = CORPORA / "DoNotAnswer" / "datasets" / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompt = row.get("question", "").strip()
                if prompt:
                    src = f"do_not_answer_{lang}"
                    records.append({
                        "prompt": prompt,
                        "lang": lang,
                        "source": src,
                        "traceid": _stableid(prompt, src),
                    })
    logger.info(f"[do_not_answer] {len(records)} records")
    return records


def load_flames() -> list[dict]:
    records = []
    path = CORPORA / "Flames" / "Flames_1k_Chinese.jsonl"
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "zh",
                    "source": "flames",
                    "traceid": _stableid(prompt, "flames"),
                })
    logger.info(f"[flames] {len(records)} records")
    return records


def load_safety_prompts(max_per_category: int = 500) -> list[dict]:
    records = []
    path = CORPORA / "SafetyPrompts" / "typical_safety_scenarios.json"
    if not path.exists():
        return records
    data = json.load(open(path, encoding="utf-8"))
    for category, items in data.items():
        for item in items[:max_per_category]:
            prompt = item.get("prompt", "").strip() if isinstance(item, dict) else str(item).strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "zh",
                    "source": "safety_prompts",
                    "traceid": _stableid(prompt, "safety_prompts"),
                })
    logger.info(f"[safety_prompts] {len(records)} records")
    return records


def load_catqa() -> list[dict]:
    records = []
    for fname, lang in [
        ("catqa_english.json", "en"),
        ("catqa_chinese.json", "zh"),
    ]:
        path = CORPORA / "CatQA" / "catqa_safetybench" / fname
        if not path.exists():
            continue
        data = json.load(open(path, encoding="utf-8"))
        src = f"catqa_{lang}"
        for category, sub in data.items():
            if isinstance(sub, dict):
                for subcategory, prompts in sub.items():
                    for p in prompts:
                        prompt = p.strip() if isinstance(p, str) else str(p).strip()
                        if prompt:
                            records.append({
                                "prompt": prompt,
                                "lang": lang,
                                "source": src,
                                "traceid": _stableid(prompt, src),
                            })
            elif isinstance(sub, list):
                for p in sub:
                    prompt = p.strip() if isinstance(p, str) else str(p).strip()
                    if prompt:
                        records.append({
                            "prompt": prompt,
                            "lang": lang,
                            "source": src,
                            "traceid": _stableid(prompt, src),
                        })
    logger.info(f"[catqa] {len(records)} records")
    return records


def load_all_datasets(seed: int = 42) -> list[dict]:
    """加载所有数据源，去重后全量返回（不截断，由调用方决定用量）。"""
    loaders = [
        load_seval_base_risk,
        load_alert,
        load_saladbench,
        load_advbench,
        load_harmbench,
        load_strongreject,
        load_malicious_instruct,
        load_forbidden_questions,
        load_do_not_answer,
        load_flames,
        lambda: load_safety_prompts(max_per_category=400),
        load_catqa,
    ]

    all_records: list[dict] = []
    seen_traceids: set[str] = set()

    for loader in loaders:
        for r in loader():
            tid = r["traceid"]
            if tid not in seen_traceids:
                seen_traceids.add(tid)
                all_records.append(r)

    logger.info(f"总计去重后: {len(all_records)} 条")

    rng = random.Random(seed)
    rng.shuffle(all_records)
    return all_records


# ---------------------------------------------------------------------------
# LLM Prompt 构建（与 build_sft_dataset_v2.py 保持一致）
# ---------------------------------------------------------------------------

def _at_list() -> str:
    return "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(ATTACK_TYPE_NAMES))


def _rc_list() -> str:
    return "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(RISK_CATEGORY_NAMES))


def _dm_list() -> str:
    return "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(DOMAIN_NAMES))


def build_classify_messages(prompt: str, lang: str) -> list[dict]:
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
        f"=== ATTACK TYPE OPTIONS ===\n{_at_list()}\n\n"
        f"=== RISK CATEGORY OPTIONS ===\n{_rc_list()}\n\n"
        f"=== DOMAIN OPTIONS ===\n{_dm_list()}\n\n"
        f"=== JAILBREAK PROMPT ===\n{prompt[:1500]}\n\n"
        "Classify this prompt. Return exactly 3 lines (Attack Type / Risk Category / Domain):"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


NOT_ATTACK_MARKER = "NOT_JAILBREAK"


def build_intent_messages(prompt: str, lang: str) -> list[dict]:
    if lang == "zh":
        system_msg = (
            "你是一名安全研究员。你的任务是判断一条文本是否为【越狱攻击提示词】，如果是，提取攻击者的【直接目的】。\n\n"
            "判断标准：越狱攻击提示词是指试图诱导 AI 模型输出有害内容的文本，包括角色扮演、情景设置、命令注入等手段，\n"
            "或者直接请求有害/非法信息（如武器制造、网络攻击、违禁内容等）。\n\n"
            "规则：\n"
            "1. 如果该文本【不是】越狱攻击提示词（例如普通问题、无害描述、日常对话），仅输出：NOT_JAILBREAK\n"
            "2. 如果是越狱攻击提示词，输出攻击者想让模型做的有害行为的【类型/类别】，用概括性语言描述\n"
            "3. 输出抽象的有害行为类型，不包含具体人名、地名、工具名、技术细节\n"
            "4. 用一句简洁的中文表达，以动词开头（如：提供…、生成…、协助…、教授…）\n"
            "5. 不要解释、不要额外说明，仅输出直接目的或 NOT_JAILBREAK"
        )
        user_msg = (
            f"以下文本是否为越狱攻击提示词？如果是，提取攻击者的直接目的；如果不是，输出 NOT_JAILBREAK。\n\n"
            f"文本：\n{prompt[:2000]}\n\n输出："
        )
    else:
        system_msg = (
            "You are a safety researcher. Your task is to determine if a text is a JAILBREAK ATTACK PROMPT, "
            "and if so, extract the attacker's DIRECT PURPOSE.\n\n"
            "A jailbreak attack prompt tries to manipulate an AI into producing harmful content, via roleplay, "
            "scenario framing, instruction injection, or direct requests for harmful/illegal information.\n\n"
            "Rules:\n"
            "1. If the text is NOT a jailbreak attack (e.g., benign question, harmless description, normal chat), "
            "output ONLY: NOT_JAILBREAK\n"
            "2. If it IS a jailbreak attack, output the TYPE/CATEGORY of harmful behavior, generalizing language\n"
            "3. Do NOT include specific names, places, tool names, or technical details\n"
            "4. Express in one concise sentence, starting with a verb (e.g., Provide..., Generate...)\n"
            "5. No explanation — output ONLY the direct purpose or NOT_JAILBREAK"
        )
        user_msg = (
            f"Is the following text a jailbreak attack prompt? "
            f"If yes, extract the attacker's direct purpose. If no, output NOT_JAILBREAK.\n\n"
            f"Text:\n{prompt[:2000]}\n\nOutput:"
        )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# 解析工具
# ---------------------------------------------------------------------------

def _fuzzy_match(raw: str, valid_names: list[str]) -> str | None:
    raw = raw.strip().strip('"').strip("'").strip(".")
    raw_lower = raw.lower()
    for name in valid_names:
        if name.lower() == raw_lower:
            return name
    for name in valid_names:
        if raw_lower in name.lower() or name.lower() in raw_lower:
            return name
    return None


def _fuzzy_match_with_abbr(raw: str, valid_names: list[str], defs: dict) -> str | None:
    result = _fuzzy_match(raw, valid_names)
    if result:
        return result
    raw_lower = raw.strip().lower()
    for name, meta in defs.items():
        if raw_lower == meta["abbr"].lower():
            return name
    return None


def parse_classification(raw: str | None) -> tuple[str | None, str | None, str | None]:
    if raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
    if not raw:
        return None, None, None
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(
            r"^(Attack Type|Risk Category|Domain)\s*[:：]\s*", "", line, flags=re.IGNORECASE
        )
        cleaned.append(line.strip())
    at = rc = dm = None
    if len(cleaned) >= 3:
        at = _fuzzy_match_with_abbr(cleaned[0], ATTACK_TYPE_NAMES, ATTACK_TYPES)
        rc = _fuzzy_match_with_abbr(cleaned[1], RISK_CATEGORY_NAMES, RISK_CATEGORIES)
        dm = _fuzzy_match_with_abbr(cleaned[2], DOMAIN_NAMES, DOMAINS)
    else:
        for line in cleaned:
            if at is None:
                at = _fuzzy_match_with_abbr(line, ATTACK_TYPE_NAMES, ATTACK_TYPES)
            if rc is None:
                rc = _fuzzy_match_with_abbr(line, RISK_CATEGORY_NAMES, RISK_CATEGORIES)
            if dm is None:
                dm = _fuzzy_match_with_abbr(line, DOMAIN_NAMES, DOMAINS)
    return at, rc, dm


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
# 断点
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done[d["traceid"]] = d
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(
    target: int = 50000,
    batch_size: int = 200,
    max_workers: int = 16,
    model_name: str | None = None,
    dry_run: bool = False,
    seed: int = 42,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> None:
    out_dir = output_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME
    output_sft = out_dir / "attack_sft_50k.json"
    output_detail = out_dir / "attack_sft_50k_detail.jsonl"

    # 1. 加载全部数据（不截断，过滤后再取 target 条）
    all_records = load_all_datasets(seed=seed)
    logger.info(f"全量加载: {len(all_records)} 条，过滤后将保留前 {target} 条")
    if limit:
        all_records = all_records[:limit]
        logger.info(f"--limit 截断至 {len(all_records)} 条")

    # 2. 获取模型
    if model_name is None:
        models = get_available_models()
        if not models:
            raise RuntimeError("No local inference service is available.")
        model_name = models[0]
    logger.info(f"使用模型: {model_name}")

    # 3. 断点续跑
    done_map = load_checkpoint(checkpoint_path)
    logger.info(f"Checkpoint 已有: {len(done_map)} 条")

    todo = [r for r in all_records if r["traceid"] not in done_map]
    logger.info(f"待处理: {len(todo)} 条")

    # --- dry-run ---
    if dry_run:
        # 找第一条实际是攻击的 prompt 来演示
        candidates = todo[:20] if todo else all_records[:20]
        record = candidates[0]
        for r in candidates:
            if len(r["prompt"]) >= 30:
                record = r
                break
        logger.info(f"=== DRY RUN: traceid={record['traceid']} source={record['source']} lang={record['lang']} ===")
        logger.info(f"prompt: {record['prompt'][:300]}")

        classify_raw = chat_completion(
            build_classify_messages(record["prompt"], record["lang"]),
            model=model_name, max_tokens=100, temperature=0.0,
        )
        at, rc, dm = parse_classification(classify_raw)
        at = at or "Compositional / Hybrid Attacks"
        rc = rc or "Illegal Wrongdoing & Criminal Enablement"
        dm = dm or "General Assistant"

        intent_raw = chat_completion(
            build_intent_messages(record["prompt"], record["lang"]),
            model=model_name, max_tokens=200, temperature=0.3,
        )
        intent_clean = re.sub(r"<think>.*?</think>", "", intent_raw or "", flags=re.DOTALL)
        intent_clean = re.sub(r"<think>.*", "", intent_clean, flags=re.DOTALL).strip()

        if intent_clean == NOT_ATTACK_MARKER or not intent_clean:
            logger.info("[DRY RUN] 该条被判定为非攻击 prompt，已过滤")
            return

        sft_entry = {
            "instruction": build_instruction(at, rc, dm),
            "input": intent_clean,
            "output": record["prompt"],
        }
        print(json.dumps(sft_entry, indent=2, ensure_ascii=False))
        return

    if not todo:
        logger.info("全部已完成，直接生成输出文件。")
    else:
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
        pbar = tqdm(total=total, unit="条", desc="标注进度", dynamic_ncols=True) if use_tqdm else None

        try:
            with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
                for batch_start in range(0, total, batch_size):
                    batch = todo[batch_start: batch_start + batch_size]

                    classify_raws = batch_chat_completion(
                        [build_classify_messages(r["prompt"], r["lang"]) for r in batch],
                        model=model_name, max_tokens=100, temperature=0.0,
                        max_workers=max_workers,
                    )
                    intent_raws = batch_chat_completion(
                        [build_intent_messages(r["prompt"], r["lang"]) for r in batch],
                        model=model_name, max_tokens=200, temperature=0.3,
                        max_workers=max_workers,
                    )

                    filtered_count = 0
                    for record, classify_raw, intent_raw in zip(batch, classify_raws, intent_raws):
                        intent_clean = re.sub(r"<think>.*?</think>", "", intent_raw or "", flags=re.DOTALL)
                        intent_clean = re.sub(r"<think>.*", "", intent_clean, flags=re.DOTALL).strip()

                        # 过滤：非攻击 prompt 或意图提取失败
                        if (not intent_clean
                                or intent_clean == NOT_ATTACK_MARKER
                                or intent_clean.upper() == "NOT_JAILBREAK"
                                or intent_clean == record["prompt"].strip()
                                or intent_clean == record["prompt"][:200].strip()):
                            failed_intent += 1
                            filtered_count += 1
                            # 仍写入 checkpoint（标记 skipped=True），避免重复处理
                            skip_entry = {
                                "traceid": record["traceid"],
                                "_lang": record["lang"],
                                "source": record["source"],
                                "_skipped": True,
                            }
                            ckpt_f.write(json.dumps(skip_entry, ensure_ascii=False) + "\n")
                            done_map[record["traceid"]] = skip_entry
                            continue

                        at, rc, dm = parse_classification(classify_raw)
                        if at is None:
                            failed_classify += 1
                            at = "Compositional / Hybrid Attacks"
                        if rc is None:
                            failed_classify += 1
                            rc = "Illegal Wrongdoing & Criminal Enablement"
                        if dm is None:
                            failed_classify += 1
                            dm = "General Assistant"

                        entry = {
                            "traceid": record["traceid"],
                            "_lang": record["lang"],
                            "source": record["source"],
                            "attack_type": at,
                            "risk_category": rc,
                            "domain": dm,
                            "input_text": intent_clean,
                            "prompt": record["prompt"],
                        }
                        ckpt_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        done_map[record["traceid"]] = entry

                    ckpt_f.flush()
                    processed += len(batch)

                    if pbar:
                        pbar.update(len(batch))
                        pbar.set_postfix(classify_fail=failed_classify, filtered=failed_intent)
                    else:
                        elapsed = time.time() - start_time
                        speed = processed / elapsed if elapsed > 0 else 0
                        eta = (total - processed) / speed if speed > 0 else float("inf")
                        kept = processed - failed_intent
                        logger.info(
                            f"  {processed}/{total} ({processed / total * 100:.1f}%) "
                            f"| kept: {kept} | filtered(non-attack): {failed_intent} "
                            f"| classify_fix: {failed_classify} "
                            f"| 速度: {speed:.1f} 条/s | ETA: {eta / 60:.1f} min"
                        )

        except KeyboardInterrupt:
            logger.info(f"中断，已完成 {processed}/{total} 条，checkpoint 已保存。")
            if pbar:
                pbar.close()
            _write_outputs(all_records, done_map, output_sft, output_detail, target=target)
            return

        if pbar:
            pbar.close()

        elapsed = time.time() - start_time
        logger.info(
            f"标注完成！{processed} 条 | classify_fail: {failed_classify} | "
            f"intent_fail: {failed_intent} | 耗时 {elapsed / 60:.1f} min"
        )

    _write_outputs(all_records, done_map, output_sft, output_detail, target=target)


def _write_outputs(
    all_records: list[dict],
    done_map: dict[str, dict],
    output_sft: Path,
    output_detail: Path,
    target: int = 50000,
) -> None:
    sft_list = []
    for record in all_records:
        if len(sft_list) >= target:
            break
        entry = done_map.get(record["traceid"])
        if entry is None or entry.get("_skipped"):
            continue
        sft_list.append({
            "instruction": build_instruction(entry["attack_type"], entry["risk_category"], entry["domain"]),
            "input": entry["input_text"],
            "output": entry["prompt"],
        })

    with open(output_sft, "w", encoding="utf-8") as f:
        json.dump(sft_list, f, ensure_ascii=False, indent=2)
    logger.info(f"SFT 数据集: {len(sft_list)} 条（目标 {target} 条）→ {output_sft}")

    detail_count = 0
    with open(output_detail, "w", encoding="utf-8") as f:
        for record in all_records:
            if detail_count >= target:
                break
            entry = done_map.get(record["traceid"])
            if entry is None or entry.get("_skipped"):
                continue
            detail_count += 1
            f.write(json.dumps({
                "traceid": entry["traceid"],
                "lang": entry.get("_lang", "en"),
                "source": entry.get("source", ""),
                "attack_type": entry["attack_type"],
                "risk_category": entry["risk_category"],
                "domain": entry["domain"],
                "instruction": build_instruction(entry["attack_type"], entry["risk_category"], entry["domain"]),
                "input": entry["input_text"],
                "output": entry["prompt"],
            }, ensure_ascii=False) + "\n")
    logger.info(f"详细版: → {output_detail}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="只处理第 1 条，预览格式")
    parser.add_argument("--target", type=int, default=50000, help="采样目标条数（默认 50000）")
    parser.add_argument("--batch-size", type=int, default=200, help="每批大小（默认 200）")
    parser.add_argument("--workers", type=int, default=16, help="并发线程数（默认 16）")
    parser.add_argument("--model", type=str, default=None, help="指定 vLLM 模型名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认 data/sft_50k/）")
    parser.add_argument("--limit", type=int, default=None, help="调试用：只处理前 N 条")
    args = parser.parse_args()

    run(
        target=args.target,
        batch_size=args.batch_size,
        max_workers=args.workers,
        model_name=args.model,
        dry_run=args.dry_run,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
