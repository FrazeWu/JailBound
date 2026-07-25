#!/usr/bin/env python3
"""
build_sft_broad.py — 从 corpora/downloaded_datasets 中筛选「文本生成类型 & 非原始越狱模板」
数据集，整合后使用 vLLM 标注（与 build_sft_50k.py 相同三维分类 + 意图提取），
输出 LLaMA-Factory Alpaca 格式 SFT 数据集。

新增数据源（均为「直接有害问题/场景」而非越狱模板）：
  AART            ~600   AI-crafted regional harmful prompts (EN)
  Aegis-2.0      ~15k   Real harmful user queries (EN, filtered unsafe)
  BeaverTails      ~14   Direct harmful questions (EN, eval set)
  HarmEval        ~316   Unsafe Q&A pairs (EN)
  MedSafetyBench  ~800   Medical harmful instructions (EN)
  TDCRedTeaming   ~100   Red-team target behaviors (EN)
  JADE            ~420   T2I harmful prompts (EN+ZH)
  CValuesResp     ~664   Chinese responsibility prompts (ZH, some benign→filtered)
  PRISM           ~800   Real user conversation starters (EN, controversial)
  DiaSafety      ~4500   Unsafe dialogue contexts (EN)
  CoSafe         ~1400   Multi-turn unsafe conversation prompts (EN)

运行：
  python build_sft_broad.py --dry-run
  python build_sft_broad.py --batch-size 200 --workers 16
  python build_sft_broad.py --limit 500 --dry-run
"""

from __future__ import annotations

import argparse
import csv
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

# ── 复用 build_sft_50k 中的分类定义、标注 prompt 构建、解析工具 ─────────────
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_b50k", _HERE / "build_sft_50k.py")
_b50k = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_b50k)

ATTACK_TYPES = _b50k.ATTACK_TYPES
RISK_CATEGORIES = _b50k.RISK_CATEGORIES
DOMAINS = _b50k.DOMAINS
ATTACK_TYPE_NAMES = _b50k.ATTACK_TYPE_NAMES
RISK_CATEGORY_NAMES = _b50k.RISK_CATEGORY_NAMES
DOMAIN_NAMES = _b50k.DOMAIN_NAMES
NOT_ATTACK_MARKER = _b50k.NOT_ATTACK_MARKER

build_classify_messages = _b50k.build_classify_messages
build_intent_messages = _b50k.build_intent_messages
parse_classification = _b50k.parse_classification
build_instruction = _b50k.build_instruction
load_checkpoint = _b50k.load_checkpoint
_stableid = _b50k._stableid


def build_intent_messages_broad(prompt: str, lang: str) -> list[dict]:
    """宽源数据集专用意图提取 prompt：跳过越狱检测，直接提取有害行为类型。

    这批数据已经过 unsafe 标签过滤，不需要再判断是否为越狱 prompt，
    直接提取攻击者/用户的有害意图类型即可。
    """
    if lang == "zh":
        system_msg = (
            "你是一名安全研究员。给定一条有害或风险内容，提取其中体现的【有害行为类型】。\n\n"
            "规则：\n"
            "1. 输出攻击者/用户想要达成的有害目的的【类型/类别】，用概括性语言描述\n"
            "2. 输出抽象的有害行为类型，不包含具体人名、地名、工具名、技术细节\n"
            "3. 用一句简洁的中文表达，以动词开头（如：提供…、生成…、协助…、教授…）\n"
            "4. 如果内容完全无害（如日常对话、技术问题），输出：BENIGN\n"
            "5. 不要解释、不要额外说明，仅输出有害行为类型或 BENIGN"
        )
        user_msg = (
            f"以下内容体现了什么有害行为？请提取有害行为类型。\n\n"
            f"内容：\n{prompt[:2000]}\n\n输出："
        )
    else:
        system_msg = (
            "You are a safety researcher. Given a harmful or risky piece of content, "
            "extract the TYPE of harmful behavior it represents.\n\n"
            "Rules:\n"
            "1. Output the TYPE/CATEGORY of harmful behavior the attacker/user wants to achieve, "
            "using generalizing language\n"
            "2. Do NOT include specific names, places, tool names, or technical details\n"
            "3. Express in one concise sentence, starting with a verb (e.g., Provide..., Generate...)\n"
            "4. If the content is completely harmless (e.g., casual chat, benign tech question), "
            "output: BENIGN\n"
            "5. No explanation — output ONLY the harmful behavior type or BENIGN"
        )
        user_msg = (
            f"What harmful behavior does the following content represent? "
            f"Extract the harmful behavior type.\n\n"
            f"Content:\n{prompt[:2000]}\n\nOutput:"
        )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CORPORA = _PROJ / "corpora" / "downloaded_datasets"
DEFAULT_OUT_DIR = _HERE / "data" / "sft_broad"
CHECKPOINT_NAME = ".build_broad_checkpoint.jsonl"


# ═══════════════════════════════════════════════════════════════════════════
# 新数据源加载器
# ═══════════════════════════════════════════════════════════════════════════

def load_aart() -> list[dict]:
    """AART: AI-crafted regional harmful prompts."""
    records: list[dict] = []
    path = CORPORA / "AART" / "data" / "aart-v1-20231117.csv"
    if not path.exists():
        logger.warning(f"[AART] not found: {path}")
        return records
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = row.get("prompt", "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "aart",
                    "traceid": _stableid(prompt, "aart"),
                })
    logger.info(f"[aart] {len(records)} records")
    return records


def load_aegis() -> list[dict]:
    """Aegis-AI-Content-Safety-Dataset-2.0: real harmful queries (unsafe only)."""
    records: list[dict] = []
    base = CORPORA / "Aegis-AI-Content-Safety-Dataset-2.0"
    for split in ("train", "validation", "test"):
        path = base / f"{split}.json"
        if not path.exists():
            continue
        data = json.load(open(path, encoding="utf-8"))
        for obj in data:
            if obj.get("prompt_label") != "unsafe":
                continue
            prompt = obj.get("prompt", "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "aegis",
                    "traceid": _stableid(prompt, "aegis"),
                })
    logger.info(f"[aegis] {len(records)} records (unsafe only)")
    return records


def load_beavertails() -> list[dict]:
    """BeaverTails evaluation set: direct harmful questions."""
    import glob as _glob
    records: list[dict] = []
    pattern = str(CORPORA / "BeaverTails" / "data" / "BeaverTails-Evaluation" / "*.json")
    for fpath in _glob.glob(pattern):
        try:
            obj = json.load(open(fpath, encoding="utf-8"))
            if isinstance(obj, dict):
                prompt = obj.get("prompt", "").strip()
                if prompt:
                    records.append({
                        "prompt": prompt,
                        "lang": "en",
                        "source": "beavertails",
                        "traceid": _stableid(prompt, "beavertails"),
                    })
        except Exception:
            pass
    logger.info(f"[beavertails] {len(records)} records")
    return records


def load_harmeval() -> list[dict]:
    """HarmEval: unsafe questions with safe answers."""
    records: list[dict] = []
    path = (
        CORPORA
        / "HarmEval"
        / "function_vectors-main"
        / "dataset_files"
        / "our_datasets"
        / "fv_unsafe_Q_safe_A.json"
    )
    if not path.exists():
        logger.warning(f"[HarmEval] not found: {path}")
        return records
    data = json.load(open(path, encoding="utf-8"))
    for obj in data:
        raw = obj.get("input", "").strip()
        # Strip "Task: Write a function that performs the task " or "Task: " prefix
        prompt = re.sub(r"^Task:\s*(Write a function that performs the task\s*)?", "", raw).strip()
        if prompt:
            records.append({
                "prompt": prompt,
                "lang": "en",
                "source": "harmeval",
                "traceid": _stableid(prompt, "harmeval"),
            })
    logger.info(f"[harmeval] {len(records)} records")
    return records


def load_medsafety() -> list[dict]:
    """MedSafetyBench: medical harmful instructions."""
    records: list[dict] = []
    base = (
        CORPORA
        / "MedSafetyBench"
        / "datasets"
        / "train-splits-used-for-ft"
        / "med_safety"
    )
    # prefer n800, fall back to n500
    for fname in ("ft_safety_med_n800.json", "ft_safety_med_n500.json"):
        path = base / fname
        if not path.exists():
            continue
        data = json.load(open(path, encoding="utf-8"))
        seen = set()
        for obj in data:
            prompt = obj.get("instruction", "").strip()
            if prompt and prompt not in seen:
                seen.add(prompt)
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "medsafety",
                    "traceid": _stableid(prompt, "medsafety"),
                })
        break  # use first found file only
    logger.info(f"[medsafety] {len(records)} records")
    return records


def load_tdc_redteam() -> list[dict]:
    """TDCRedTeaming: red-team target behavior strings."""
    records: list[dict] = []
    base = CORPORA / "TDCRedTeaming" / "red_teaming" / "data"
    for split in ("dev", "test"):
        path = base / split / "behaviors.json"
        if not path.exists():
            continue
        behaviors = json.load(open(path, encoding="utf-8"))
        for b in behaviors:
            prompt = str(b).strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "tdc_redteam",
                    "traceid": _stableid(prompt, "tdc_redteam"),
                })
    logger.info(f"[tdc_redteam] {len(records)} records")
    return records


def load_jade() -> list[dict]:
    """JADE: T2I harmful prompts (EN + ZH)."""
    records: list[dict] = []
    base = CORPORA / "JADE"
    for version in ("jade-t2i-v1.0", "jade-t2i-v2.0"):
        vdir = base / version
        if not vdir.exists():
            continue
        for fname in vdir.glob("*.csv"):
            lang = "zh" if "_zh" in fname.stem else "en"
            with open(fname, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    prompt = (row.get("提示词") or row.get("prompt") or "").strip()
                    if prompt:
                        records.append({
                            "prompt": prompt,
                            "lang": lang,
                            "source": "jade",
                            "traceid": _stableid(prompt, "jade"),
                        })
    logger.info(f"[jade] {len(records)} records")
    return records


def load_cvalues_prompts() -> list[dict]:
    """CValuesResponsibilityPrompts: Chinese responsibility prompts (LLM will filter benign)."""
    records: list[dict] = []
    path = (
        CORPORA
        / "CValuesResponsibilityPrompts"
        / "dataset"
        / "cvalues_responsibility_prompts.jsonl"
    )
    if not path.exists():
        logger.warning(f"[cvalues] not found: {path}")
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
                    "source": "cvalues_prompts",
                    "traceid": _stableid(prompt, "cvalues_prompts"),
                })
    logger.info(f"[cvalues_prompts] {len(records)} records")
    return records


def load_prism(max_records: int = 2000) -> list[dict]:
    """PRISM: real user conversation opening prompts (controversial topics)."""
    records: list[dict] = []
    path = CORPORA / "PRISM" / "data" / "conversations.jsonl"
    if not path.exists():
        logger.warning(f"[prism] not found: {path}")
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(records) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("opening_prompt", "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "prism",
                    "traceid": _stableid(prompt, "prism"),
                })
    logger.info(f"[prism] {len(records)} records")
    return records


def load_diasafety() -> list[dict]:
    """DiaSafety: unsafe dialogue user contexts."""
    records: list[dict] = []
    base = CORPORA / "DiaSafety" / "DiaSafety_dataset"
    for split in ("train", "test", "val"):
        path = base / f"{split}.json"
        if not path.exists():
            continue
        data = json.load(open(path, encoding="utf-8"))
        for obj in data:
            if obj.get("label") != "Unsafe":
                continue
            prompt = obj.get("context", "").strip()
            if prompt:
                records.append({
                    "prompt": prompt,
                    "lang": "en",
                    "source": "diasafety",
                    "traceid": _stableid(prompt, "diasafety"),
                })
    logger.info(f"[diasafety] {len(records)} records (unsafe label only)")
    return records


def load_cosafe() -> list[dict]:
    """CoSafe: last user turn from each multi-turn unsafe conversation."""
    import os
    records: list[dict] = []
    base = CORPORA / "CoSafe" / "CoSafe datasets"
    if not base.exists():
        logger.warning(f"[cosafe] not found: {base}")
        return records
    for fn in os.listdir(base):
        fpath = base / fn
        if not fn.endswith(".json"):
            continue
        category = fn.replace(".json", "").replace(",", "/")
        with open(fpath, encoding="utf-8") as f:
            content = f.read().strip()
        # each file is a JSON list of conversations
        try:
            convs = json.loads(content)
        except json.JSONDecodeError:
            # try jsonl
            convs = [json.loads(l) for l in content.split("\n") if l.strip()]
        for conv in convs:
            if not isinstance(conv, list):
                continue
            # extract all user messages
            user_msgs = [
                turn.get("content", "").strip()
                for turn in conv
                if isinstance(turn, dict) and turn.get("role") == "user"
            ]
            # take the last user message (usually the one probing boundaries)
            if user_msgs:
                prompt = user_msgs[-1]
                if prompt and len(prompt) > 5:
                    records.append({
                        "prompt": prompt,
                        "lang": "en",
                        "source": f"cosafe_{category.replace(' ','_')}",
                        "traceid": _stableid(prompt, "cosafe"),
                    })
    logger.info(f"[cosafe] {len(records)} records")
    return records


# ═══════════════════════════════════════════════════════════════════════════
# 整合入口
# ═══════════════════════════════════════════════════════════════════════════

def load_broad_datasets(seed: int = 42) -> list[dict]:
    """加载所有宽源数据集，去重后返回（不截断）。"""
    loaders = [
        load_aart,
        load_aegis,
        load_beavertails,
        load_harmeval,
        load_medsafety,
        load_tdc_redteam,
        load_jade,
        load_cvalues_prompts,
        load_prism,
        load_diasafety,
        load_cosafe,
    ]

    all_records: list[dict] = []
    seen_traceids: set[str] = set()

    for loader in loaders:
        for r in loader():
            if len(r.get("prompt", "")) < 5:
                continue
            tid = r["traceid"]
            if tid not in seen_traceids:
                seen_traceids.add(tid)
                all_records.append(r)

    logger.info(f"宽源去重后合计: {len(all_records)} 条")

    rng = random.Random(seed)
    rng.shuffle(all_records)
    return all_records


# ═══════════════════════════════════════════════════════════════════════════
# 输出写入
# ═══════════════════════════════════════════════════════════════════════════

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
            "instruction": build_instruction(
                entry["attack_type"], entry["risk_category"], entry["domain"]
            ),
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
            f.write(
                json.dumps(
                    {
                        "traceid": entry["traceid"],
                        "lang": entry.get("_lang", "en"),
                        "source": entry.get("source", ""),
                        "attack_type": entry["attack_type"],
                        "risk_category": entry["risk_category"],
                        "domain": entry["domain"],
                        "instruction": build_instruction(
                            entry["attack_type"],
                            entry["risk_category"],
                            entry["domain"],
                        ),
                        "input": entry["input_text"],
                        "output": entry["prompt"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info(f"详细版: → {output_detail}")


# ═══════════════════════════════════════════════════════════════════════════
# 主运行逻辑
# ═══════════════════════════════════════════════════════════════════════════

def run(
    target: int = 50000,
    batch_size: int = 200,
    max_workers: int = 16,
    model_name: str | None = None,
    dry_run: bool = False,
    seed: int = 42,
    output_dir: Path | None = None,
    limit: int | None = None,
    export_only: bool = False,
) -> None:
    out_dir = output_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME
    output_sft = out_dir / "attack_sft_broad.json"
    output_detail = out_dir / "attack_sft_broad_detail.jsonl"
    raw_output = out_dir / "raw_prompts.jsonl"

    # 1. 加载全量数据
    all_records = load_broad_datasets(seed=seed)
    logger.info(f"全量加载: {len(all_records)} 条，过滤后将保留前 {target} 条")

    if limit:
        all_records = all_records[:limit]
        logger.info(f"--limit 截断至 {len(all_records)} 条")

    # 导出原始 JSONL（供检查）
    with open(raw_output, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"原始 prompts 已写入: {raw_output}")

    if export_only:
        logger.info("--export-only 模式：仅导出原始 prompts，跳过标注。")
        return

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
        candidates = todo[:20] if todo else all_records[:20]
        record = candidates[0]
        for r in candidates:
            if len(r["prompt"]) >= 30:
                record = r
                break
        logger.info(
            f"=== DRY RUN: traceid={record['traceid']} source={record['source']} lang={record['lang']} ==="
        )
        logger.info(f"prompt: {record['prompt'][:300]}")

        classify_raw = chat_completion(
            build_classify_messages(record["prompt"], record["lang"]),
            model=model_name,
            max_tokens=100,
            temperature=0.0,
        )
        at, rc, dm = parse_classification(classify_raw)
        at = at or "Compositional / Hybrid Attacks"
        rc = rc or "Illegal Wrongdoing & Criminal Enablement"
        dm = dm or "General Assistant"

        intent_raw = chat_completion(
            build_intent_messages_broad(record["prompt"], record["lang"]),
            model=model_name,
            max_tokens=600,
            temperature=0.3,
        )
        intent_clean = re.sub(r"<think>.*?</think>", "", intent_raw or "", flags=re.DOTALL)
        intent_clean = re.sub(r"<think>.*", "", intent_clean, flags=re.DOTALL).strip()

        if not intent_clean or intent_clean.upper() == "BENIGN":
            logger.info("[DRY RUN] 该条被判定为无害内容，已过滤")
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
                    batch = todo[batch_start : batch_start + batch_size]

                    classify_raws = batch_chat_completion(
                        [build_classify_messages(r["prompt"], r["lang"]) for r in batch],
                        model=model_name,
                        max_tokens=100,
                        temperature=0.0,
                        max_workers=max_workers,
                    )
                    intent_raws = batch_chat_completion(
                        [build_intent_messages_broad(r["prompt"], r["lang"]) for r in batch],
                        model=model_name,
                        max_tokens=600,
                        temperature=0.3,
                        max_workers=max_workers,
                    )

                    for record, classify_raw, intent_raw in zip(batch, classify_raws, intent_raws):
                        intent_clean = re.sub(
                            r"<think>.*?</think>", "", intent_raw or "", flags=re.DOTALL
                        )
                        intent_clean = re.sub(
                            r"<think>.*", "", intent_clean, flags=re.DOTALL
                        ).strip()

                        is_skipped = (
                            not intent_clean
                            or intent_clean.upper() == "BENIGN"
                            or intent_clean == record["prompt"].strip()
                            or intent_clean == record["prompt"][:200].strip()
                        )

                        if is_skipped:
                            failed_intent += 1
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


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="只处理第 1 条，预览格式")
    parser.add_argument("--export-only", action="store_true", help="仅导出原始 prompts JSONL，不标注")
    parser.add_argument("--target", type=int, default=50000, help="保留目标条数（默认 50000）")
    parser.add_argument("--batch-size", type=int, default=200, help="每批大小（默认 200）")
    parser.add_argument("--workers", type=int, default=16, help="并发线程数（默认 16）")
    parser.add_argument("--model", type=str, default=None, help="指定 vLLM 模型名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="输出目录（默认 data/sft_broad/）"
    )
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
        export_only=args.export_only,
    )


if __name__ == "__main__":
    main()
