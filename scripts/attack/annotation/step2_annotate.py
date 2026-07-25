#!/usr/bin/env python3
"""
step2_annotate.py — Step 2 独立脚本：断点重传 + tqdm 进度条

功能：
  对 reverse_engineered_en.jsonl（100K 条）进行 domain classification 标注。
  支持：
    - 批量处理（每批 BATCH_SIZE 条），每批完成后立即写入 checkpoint
    - 断点续跑：已处理条目自动跳过
    - tqdm 进度条（含速度、ETA）
    - Ctrl+C 安全退出（保留已写入的 checkpoint）

用法：
    python step2_annotate.py [--input PATH] [--checkpoint PATH] [--output PATH]
                             [--batch-size N] [--workers N]

示例：
    # 默认跑 EN 数据，batch=500，workers=16
    python step2_annotate.py

    # 恢复中断的任务（自动从 checkpoint 续跑）
    python step2_annotate.py --batch-size 500 --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent

from s_eval.vllm_client import batch_chat_completion, get_available_models  # noqa: E402
from annotate import (  # noqa: E402
    RISK_TO_THREAT,
    CATEGORY_TO_JAILBREAK,
    build_domain_messages,
    parse_domain,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
DATA_DIR = _HERE / "data" / "full_pipeline"
DEFAULT_INPUT = DATA_DIR / "reverse_engineered_en.jsonl"
DEFAULT_CHECKPOINT = DATA_DIR / ".ann_checkpoint_en.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "annotated_en.jsonl"


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_checkpoint(checkpoint_path: Path) -> dict[str, dict]:
    """加载已完成的标注，返回 {traceid: annotated_record}"""
    done: dict[str, dict] = {}
    if not checkpoint_path.exists():
        return done
    with open(checkpoint_path, encoding="utf-8") as f:
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


def merge_to_output(
    original_records: list[dict],
    done_map: dict[str, dict],
    output_path: Path,
) -> int:
    """按原始顺序合并标注结果到输出文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in original_records:
            tid = r["traceid"]
            if tid in done_map:
                f.write(json.dumps(done_map[tid], ensure_ascii=False) + "\n")
                written += 1
    return written


def run_annotation(
    input_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 500,
    max_workers: int = 16,
    model_name: str | None = None,
) -> None:
    # 1. 加载输入
    logger.info(f"加载输入: {input_path}")
    records = load_jsonl(input_path)
    logger.info(f"  共 {len(records)} 条记录")

    # 2. 加载 checkpoint（断点续跑）
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    done_map = load_checkpoint(checkpoint_path)
    logger.info(f"  已完成: {len(done_map)} 条（从 checkpoint 恢复）")

    todo = [r for r in records if r["traceid"] not in done_map]
    logger.info(f"  待处理: {len(todo)} 条")

    if not todo:
        logger.info("全部已完成，直接合并输出。")
        written = merge_to_output(records, done_map, output_path)
        logger.info(f"  输出: {written} 条 → {output_path}")
        return

    # 3. 获取模型
    if model_name is None:
        models = get_available_models()
        if not models:
            raise RuntimeError("vLLM 不可达，请检查服务状态")
        model_name = models[0]
    logger.info(f"  使用模型: {model_name}")

    # 4. 尝试导入 tqdm
    try:
        from tqdm import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False
        logger.warning("tqdm 未安装，使用日志进度。pip install tqdm 可启用进度条。")

    # 5. 批量处理
    total = len(todo)
    processed = 0
    start_time = time.time()

    pbar = (
        tqdm(total=total, unit="条", desc="Step2标注", dynamic_ncols=True)
        if use_tqdm
        else None
    )

    try:
        with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
            for batch_start in range(0, total, batch_size):
                batch = todo[batch_start : batch_start + batch_size]

                # 构建消息（用 output 字段——原始 AP prompt）
                messages_list = [build_domain_messages(r["output"]) for r in batch]

                # 并发调用 vLLM
                domain_results = batch_chat_completion(
                    messages_list,
                    model=model_name,
                    max_tokens=8,
                    temperature=0.0,
                    max_workers=max_workers,
                )

                # 写入 checkpoint
                for record, domain_raw in zip(batch, domain_results):
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
                    done_map[record["traceid"]] = annotated

                ckpt_f.flush()
                processed += len(batch)

                if pbar:
                    pbar.update(len(batch))
                else:
                    elapsed = time.time() - start_time
                    speed = processed / elapsed if elapsed > 0 else 0
                    eta = (total - processed) / speed if speed > 0 else float("inf")
                    logger.info(
                        f"  进度: {processed}/{total} ({processed / total * 100:.1f}%) "
                        f"| 速度: {speed:.1f} 条/s | ETA: {eta / 60:.1f} min"
                    )

    except KeyboardInterrupt:
        logger.info(f"\n用户中断。已完成 {processed}/{total} 条，checkpoint 已保存。")
        if pbar:
            pbar.close()
        return

    if pbar:
        pbar.close()

    elapsed = time.time() - start_time
    logger.info(f"标注完成！共 {processed} 条，耗时 {elapsed / 60:.1f} min")

    # 6. 合并输出
    written = merge_to_output(records, done_map, output_path)
    logger.info(f"输出: {written} 条 → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: domain classification 标注（断点续跑+进度条）"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="输入 JSONL 文件"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint 文件路径",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSONL 文件"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="每批处理条数（默认 500）"
    )
    parser.add_argument("--workers", type=int, default=16, help="并发线程数（默认 16）")
    parser.add_argument(
        "--model", type=str, default=None, help="指定模型名（默认自动探测）"
    )
    args = parser.parse_args()

    run_annotation(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        max_workers=args.workers,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
