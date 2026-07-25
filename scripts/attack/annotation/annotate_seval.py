#!/usr/bin/env python3
"""
annotate_seval.py — 对 S-Eval 全量攻击数据做三维标注 + 恶意意图提取，
输出 LLaMA-Factory Alpaca 格式 SFT 数据集。

标注字段:
  threat_category  : 12 类威胁缩写 (lookup from risk_type)
  jailbreak_type   : 9 种攻击类型名称 (lookup from category)
  domain           : 10 个部署领域缩写 (LLM 推断, max_tokens=8)
  malicious_intent : 从 prompt 中提取的核心恶意意图 (LLM 推断, max_tokens=80)

输出格式 (Alpaca):
  {"instruction": "<meta_prompt>", "input": "", "output": "<attack_prompt>",
   "threat_category": "...", "jailbreak_type": "...", "domain": "...",
   "malicious_intent": "...", "traceid": "..."}

用法:
  # 随机采样 100k 条 (EN+ZH 各 50k)
  python annotate_seval.py --sample 100000 --workers 32 --batch-size 500

  # 断点续跑
  python annotate_seval.py --sample 100000 --workers 32 --batch-size 500

  # dry-run 验证 5 条
  python annotate_seval.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
S_EVAL_FULL = PROJECT_ROOT / "src/generator/s_eval/full"
OUTPUT_DIR = PROJECT_ROOT / "data/sft/seval_annotated"
CHECKPOINT_FILE = OUTPUT_DIR / ".annotate_checkpoint.jsonl"
OUTPUT_FILE = OUTPUT_DIR / "seval_sft_100k.jsonl"

VLLM_BASE_URL = os.environ.get("BENCHMARK_API_BASE_URL", "http://localhost:8000")
MAX_RETRIES = 3
RETRY_DELAY = 2

# ---------------------------------------------------------------------------
# Lookup tables (no LLM needed)
# ---------------------------------------------------------------------------
RISK_TO_THREAT: dict[str, str] = {
    "Crimes_and_Illegal_Activities": "illegal_criminal",
    "Cybersecurity": "cybersecurity_misuse",
    "Data_Privacy": "privacy_personal_data",
    "Ethics_and_Morality": "unsafe_unethical",
    "Extremism": "illegal_criminal",
    "Hate_Speech": "discrimination_toxicity",
    "Inappropriate_Suggestions": "high_stakes_advice",
    "Physical_and_Mental_Health": "human_chatbot_harm",
}

CATEGORY_TO_JAILBREAK: dict[str, str] = {
    "positive_induction":        "Prefix Injection",
    "reverse_induction":         "Persuasion & Deception",
    "goal_hijacking":            "Persuasion & Deception",
    "compositional_instructions":"Scenario Nesting",
    "instruction_jailbreak":     "Contextual Demonstration",
    "deepinception":             "Scenario Nesting",
    "in_context_attack":         "Contextual Demonstration",
    "chain_of_utterances":       "Compositional / Hybrid Attacks",
    "code_injection":            "Formal Language Exploitation",
    "instruction_encryption":    "Obfuscation & Encryption",
}

DOMAIN_ABBRS = ["LE", "MED", "FIN", "LAW", "EDU", "SCI", "ENT", "SOC", "UGC", "GEN"]

DOMAIN_FULL = {
    "LE": "Law Enforcement", "MED": "Healthcare/Medicine",
    "FIN": "Finance", "LAW": "Legal",
    "EDU": "Education", "SCI": "Science/Research",
    "ENT": "Entertainment", "SOC": "Social Media",
    "UGC": "User-Generated Content", "GEN": "General",
}

# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------
def get_model(base_url: str = VLLM_BASE_URL) -> str | None:
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        models = r.json().get("data", [])
        return models[0]["id"] if models else None
    except Exception as e:
        logger.error("Cannot reach vLLM: %s", e)
        return None


def _chat_once(messages: list, model: str, max_tokens: int,
               temperature: float, base_url: str) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{base_url}/v1/chat/completions",
                      json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def chat(messages: list, model: str, max_tokens: int = 16,
         temperature: float = 0.0, base_url: str = VLLM_BASE_URL) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            return _chat_once(messages, model, max_tokens, temperature, base_url)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.warning("chat failed after %d attempts: %s", MAX_RETRIES, e)
                return ""


def batch_chat(items: list[dict], model: str, max_workers: int,
               base_url: str = VLLM_BASE_URL) -> list[str]:
    """items: list of {messages, max_tokens, temperature}"""
    results = [""] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                chat,
                item["messages"], model,
                item.get("max_tokens", 16),
                item.get("temperature", 0.0),
                base_url,
            ): idx
            for idx, item in enumerate(items)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                logger.warning("item %d failed: %s", idx, e)
            done += 1
            if done % 500 == 0:
                logger.info("  batch_chat progress: %d / %d", done, len(items))
    return results


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
DOMAIN_SYSTEM = (
    "You are a domain classifier. Given a prompt, reply with ONLY the single "
    "domain abbreviation from this list: " + ", ".join(DOMAIN_ABBRS) + ".\n"
    "Definitions: " +
    "; ".join(f"{k}={v}" for k, v in DOMAIN_FULL.items()) + "."
)

INTENT_SYSTEM = (
    "You are a security researcher. Given a jailbreak attack prompt, extract "
    "the core malicious intent in ONE concise sentence (max 20 words). "
    "Reply with ONLY the intent sentence, no preamble."
)


def build_domain_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": DOMAIN_SYSTEM},
        {"role": "user", "content": f"Prompt: {prompt[:400]}"},
    ]


def build_intent_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": f"Prompt: {prompt[:600]}"},
    ]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_domain(raw: str) -> str:
    raw = raw.strip().upper()
    for abbr in DOMAIN_ABBRS:
        if abbr in raw:
            return abbr
    return "GEN"


def parse_intent(raw: str) -> str:
    raw = raw.strip()
    # remove leading bullets / numbers
    raw = re.sub(r"^[\-\*\d\.\s]+", "", raw)
    return raw[:200] if raw else "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_s_eval_attack(sample: int, seed: int = 42) -> list[dict]:
    """Load and randomly sample from EN+ZH full attack sets."""
    files = [
        S_EVAL_FULL / "S-Eval_attack_en_full.jsonl",
        S_EVAL_FULL / "S-Eval_attack_zh_full.jsonl",
    ]
    all_records: list[dict] = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ext = json.loads(d.get("ext", "{}"))
                all_records.append({
                    "traceid":   d["traceid"],
                    "risk_type": d["risk_type"],
                    "prompt":    d["prompt"],
                    "category":  ext.get("category", ""),
                    "lang":      "en" if "en_full" in str(fpath) else "zh",
                })
    logger.info("Loaded %d total records from S-Eval full attack sets", len(all_records))

    rng = random.Random(seed)
    if sample and sample < len(all_records):
        all_records = rng.sample(all_records, sample)
        logger.info("Sampled %d records (seed=%d)", len(all_records), seed)
    return all_records


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    done[d["traceid"]] = d
                except (json.JSONDecodeError, KeyError):
                    pass
    logger.info("Checkpoint: %d already annotated", len(done))
    return done


def append_checkpoint(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core annotation
# ---------------------------------------------------------------------------
def annotate_batch(
    records: list[dict],
    model: str,
    max_workers: int,
    base_url: str,
) -> list[dict]:
    """Annotate a batch: domain + malicious_intent via LLM, others via lookup."""
    # Build two sets of requests
    domain_items = [
        {"messages": build_domain_messages(r["prompt"]), "max_tokens": 8, "temperature": 0.0}
        for r in records
    ]
    intent_items = [
        {"messages": build_intent_messages(r["prompt"]), "max_tokens": 80, "temperature": 0.0}
        for r in records
    ]

    # Fire both in parallel (interleaved via ThreadPool inside batch_chat)
    logger.info("  Running domain classification for %d records...", len(records))
    domain_results = batch_chat(domain_items, model, max_workers, base_url)
    logger.info("  Running intent extraction for %d records...", len(records))
    intent_results = batch_chat(intent_items, model, max_workers, base_url)

    annotated: list[dict] = []
    for r, dom_raw, int_raw in zip(records, domain_results, intent_results):
        annotated.append({
            "instruction":    r["prompt"],   # Alpaca instruction = attack prompt
            "input":          "",
            "output":         "",            # filled in downstream by generator
            "traceid":        r["traceid"],
            "lang":           r["lang"],
            "threat_category": RISK_TO_THREAT.get(r["risk_type"], "unsafe_unethical"),
            "jailbreak_type":  CATEGORY_TO_JAILBREAK.get(r["category"], "Persuasion & Deception"),
            "domain":          parse_domain(dom_raw),
            "malicious_intent": parse_intent(int_raw),
            "risk_type":       r["risk_type"],
            "category":        r["category"],
        })
    return annotated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    # Wait for vLLM to be ready
    base_url = args.vllm_url
    logger.info("Checking vLLM at %s ...", base_url)
    model = None
    for _ in range(30):
        model = get_model(base_url)
        if model:
            break
        logger.info("  vLLM not ready, retrying in 10s...")
        time.sleep(10)
    if not model:
        logger.error("vLLM not available after 300s. Exiting.")
        sys.exit(1)
    logger.info("Using model: %s", model)

    # Load data
    records = load_s_eval_attack(sample=args.sample, seed=args.seed)

    if args.dry_run:
        records = records[:5]
        logger.info("DRY RUN: only 5 records")

    # Load checkpoint
    done = load_checkpoint(CHECKPOINT_FILE)
    todo = [r for r in records if r["traceid"] not in done]
    logger.info("Todo: %d (already done: %d)", len(todo), len(done))

    if not todo:
        logger.info("All records already annotated.")
    else:
        # Process in batches
        batch_size = args.batch_size
        total_batches = (len(todo) + batch_size - 1) // batch_size
        for i in range(0, len(todo), batch_size):
            batch = todo[i: i + batch_size]
            batch_num = i // batch_size + 1
            logger.info("Batch %d / %d  (%d records)...", batch_num, total_batches, len(batch))
            t0 = time.time()
            annotated = annotate_batch(batch, model, args.workers, base_url)
            elapsed = time.time() - t0
            logger.info(
                "  Done in %.1fs (%.1f rec/s)", elapsed, len(batch) / max(elapsed, 1)
            )
            append_checkpoint(CHECKPOINT_FILE, annotated)
            done.update({r["traceid"]: r for r in annotated})

    # Write final output in original sample order
    logger.info("Writing final output to %s ...", OUTPUT_FILE)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            entry = done.get(r["traceid"])
            if entry:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
    logger.info("Done. Wrote %d records to %s", written, OUTPUT_FILE)

    # Print summary stats
    from collections import Counter
    threat_counts = Counter(done[r["traceid"]]["threat_category"] for r in records if r["traceid"] in done)
    jailbreak_counts = Counter(done[r["traceid"]]["jailbreak_type"] for r in records if r["traceid"] in done)
    domain_counts = Counter(done[r["traceid"]]["domain"] for r in records if r["traceid"] in done)
    logger.info("=== Threat category distribution ===")
    for k, v in threat_counts.most_common():
        logger.info("  %s: %d (%.1f%%)", k, v, 100 * v / written)
    logger.info("=== Jailbreak type distribution ===")
    for k, v in jailbreak_counts.most_common():
        logger.info("  %-35s: %d", k, v)
    logger.info("=== Domain distribution ===")
    for k, v in domain_counts.most_common():
        logger.info("  %s: %d", k, v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate S-Eval attack data for SFT.")
    p.add_argument("--sample",    type=int, default=100000, help="Number of records to sample")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--workers",   type=int, default=32,  help="Concurrent vLLM workers")
    p.add_argument("--batch-size",type=int, default=500, help="Checkpoint batch size")
    p.add_argument("--vllm-url", type=str, default=VLLM_BASE_URL)
    p.add_argument("--dry-run",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
