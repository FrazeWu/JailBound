"""
annotate.py — 为 reverse_engineered_dataset.jsonl 中的每条数据打上新分类标签

输入: s_eval/output/reverse_engineered_dataset.jsonl (5,238 条)
输出: data/annotated_dataset.jsonl

标注三个字段:
  - threat_category (abbr): 来自 category.md 的 12 类威胁 — 通过查表
  - jailbreak_type  (name): 来自 jailbreak_type.md 的 9 种攻击类型 — 通过查表
  - domain          (abbr): 来自 domain.md 的 10 个部署领域 — 通过 vLLM 推断

支持断点续跑：每 100 条写入 checkpoint，已处理条目自动跳过。

用法:
    python annotate.py [--input PATH] [--output PATH] [--workers N] [--max-tokens N]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
from vllm_client import batch_chat_completion, get_available_models  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 查表：S-Eval risk_type → 新 threat_category 缩写
# ---------------------------------------------------------------------------
RISK_TO_THREAT: dict[str, str] = {
    "Crimes_and_Illegal_Activities": "illegal_criminal",
    "Cybersecurity": "cybersecurity_misuse",
    "Data_Privacy": "privacy_personal_data",
    "Ethics_and_Morality": "unsafe_unethical",
    "Extremism": "illegal_criminal",  # 恐怖主义属于 IL 子类
    "Hate_Speech": "discrimination_toxicity",
    "Inappropriate_Suggestions": "high_stakes_advice",
    "Physical_and_Mental_Health": "human_chatbot_harm",
}

# ---------------------------------------------------------------------------
# 查表：S-Eval category → 新 jailbreak_type 名称
# ---------------------------------------------------------------------------
CATEGORY_TO_JAILBREAK: dict[str, str] = {
    "positive_induction": "Prefix Injection",
    "reverse_induction": "Persuasion & Deception",
    "goal_hijacking": "Persuasion & Deception",
    "compositional_instructions": "Scenario Nesting",
    "instruction_jailbreak": "Contextual Demonstration",
    "deepinception": "Scenario Nesting",
    "in_context_attack": "Contextual Demonstration",
    "chain_of_utterances": "Compositional / Hybrid Attacks",
    "code_injection": "Formal Language Exploitation",
    "instruction_encryption": "Obfuscation & Encryption",
}

# ---------------------------------------------------------------------------
# 领域列表（vLLM 分类用）
# ---------------------------------------------------------------------------
DOMAIN_ABBRS: list[str] = [
    "LE",
    "MED",
    "FIN",
    "LAW",
    "EDU",
    "SCI",
    "ENT",
    "SOC",
    "UGC",
    "DEV",
]
DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "LE": "Public Security / Law Enforcement — policing, investigations, public safety",
    "MED": "Healthcare / Medicine — health info, triage, treatment guidance",
    "FIN": "Finance / Payments — banking, investing, anti-scam, risk control",
    "LAW": "Legal / Government Services — legal advice, policy interpretation, public services",
    "EDU": "Education — tutoring, learning support, involving minors",
    "SCI": "Science / Research — literature synthesis, experimental design, data analysis",
    "ENT": "Enterprise Knowledge Base / RAG — internal Q&A, document retrieval, workflows",
    "SOC": "Security Operations / Cybersecurity — incident response, threat analysis, monitoring",
    "UGC": "Content Platforms / Community Moderation / News / Media — content review, moderation",
    "DEV": "Developer Tools / Code — coding assistants, agentic tooling, automation",
}

_DOMAIN_LIST_STR = "\n".join(
    f"{abbr} - {desc}" for abbr, desc in DOMAIN_DESCRIPTIONS.items()
)

DOMAIN_SYSTEM_PROMPT = (
    "You are a domain classifier for AI safety research. "
    "Given a jailbreak attack prompt, classify it into exactly ONE of the following 10 deployment domains "
    "based on which real-world AI assistant deployment context is most likely being targeted:\n\n"
    f"{_DOMAIN_LIST_STR}\n\n"
    "Respond with ONLY the 2-3 letter abbreviation (e.g., 'MED' or 'DEV'). No explanation, no punctuation."
)

DOMAIN_USER_TEMPLATE = "Attack prompt:\n{ap_text}\n\nDomain abbreviation:"


def build_domain_messages(ap_text: str) -> list[dict[str, str]]:
    """构建 vLLM 领域分类请求消息。"""
    return [
        {"role": "system", "content": DOMAIN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": DOMAIN_USER_TEMPLATE.format(ap_text=ap_text[:1200]),
        },
    ]


def parse_domain(raw: str | None) -> str:
    """从 LLM 输出中提取合法的领域缩写，失败则返回 'ENT'（兜底）。"""
    if raw is None:
        return "ENT"
    token = raw.strip().upper().split()[0].rstrip(".,;:")
    return token if token in DOMAIN_ABBRS else "ENT"


# ---------------------------------------------------------------------------
# 断点续跑辅助
# ---------------------------------------------------------------------------
def load_done_ids(checkpoint_path: Path) -> set[str]:
    """读取已处理的 traceid 集合。"""
    done: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    done.add(d["traceid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def annotate(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    max_workers: int = 8,
    max_tokens: int = 8,
) -> None:
    # 读取数据
    records: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), input_path)

    # 已完成的 traceid
    done_ids = load_done_ids(checkpoint_path)
    logger.info("Skipping %d already-annotated entries", len(done_ids))

    # 过滤待处理条目
    todo = [r for r in records if r["traceid"] not in done_ids]
    logger.info("%d entries to annotate", len(todo))

    if not todo:
        logger.info("Nothing to do. Merging checkpoint → output.")
        _merge_to_output(checkpoint_path, output_path, records)
        return

    # 自动选模型
    models = get_available_models()
    if not models:
        raise RuntimeError(
            "No local inference service is available."
        )
    model_name = models[0]
    logger.info("Using vLLM model: %s", model_name)

    # 批量分类领域（并发）
    messages_list = [build_domain_messages(r["output"]) for r in todo]
    logger.info(
        "Sending %d domain-classification requests (workers=%d) …",
        len(todo),
        max_workers,
    )

    completed = [0]

    def progress(done: int, total: int) -> None:
        completed[0] = done
        logger.info("  domain annotation progress: %d / %d", done, total)

    domain_results = batch_chat_completion(
        messages_list,
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.0,
        max_workers=max_workers,
        progress_callback=progress,
    )

    # 写入 checkpoint（追加）
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(checkpoint_path, "a") as ckpt_f:
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

            if written % 100 == 0:
                logger.info("  flushed %d entries to checkpoint", written)

    logger.info("Wrote %d annotated entries to checkpoint %s", written, checkpoint_path)

    # 合并全部已完成条目 → output（按原始顺序）
    _merge_to_output(checkpoint_path, output_path, records)


def _merge_to_output(
    checkpoint_path: Path, output_path: Path, original_records: list[dict]
) -> None:
    """按原始顺序将 checkpoint 中的已标注条目写入最终输出文件。"""
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
            else:
                logger.warning("traceid %s not annotated, skipping in output", tid)

    logger.info("Final output: %d entries → %s", written, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate reverse-engineered dataset with new taxonomy labels."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent
        / "s_eval"
        / "output"
        / "reverse_engineered_dataset.jsonl",
        help="Input JSONL path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "annotated_dataset.jsonl",
        help="Output JSONL path",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "data" / ".annotate_checkpoint.jsonl",
        help="Checkpoint JSONL path (for resume)",
    )
    p.add_argument("--workers", type=int, default=8, help="vLLM concurrent workers")
    p.add_argument(
        "--max-tokens", type=int, default=8, help="Max tokens for domain classification"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    annotate(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        max_workers=args.workers,
        max_tokens=args.max_tokens,
    )
