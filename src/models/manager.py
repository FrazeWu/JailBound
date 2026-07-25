#!/usr/bin/env python3
"""Unified model manager: check completeness and download missing/incomplete models.

Source of truth for the model list is ``benchmark/modellist.txt``. We restrict
to open-source models below 70B parameters and, per family, only keep entries
that differ in parameter scale.

Usage:
  python benchmark/manage_models.py check
  python benchmark/manage_models.py download                 # all incomplete
  python benchmark/manage_models.py download --only qwen1.5-7b-chat zephyr-7b-beta
  python benchmark/manage_models.py download --include-ok    # also re-verify completes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("MODEL_STORAGE_DIR", "models"))
HF_MIRROR = "https://hf-mirror.com"

# Open-source models <70B from modellist.txt.
# Per family we keep distinct parameter scales only.
MODELS: dict[str, str] = {
    # Llama (skip 70B / 405B)
    "llama-2-7b-chat":           "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b-chat":          "meta-llama/Llama-2-13b-chat-hf",
    "llama-3-8b-instruct":       "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3.1-8b-instruct":     "meta-llama/Llama-3.1-8B-Instruct",
    # Gemma
    "gemma-2b-it":               "google/gemma-2b-it",
    "gemma-7b-it":               "google/gemma-7b-it",
    "gemma-2-9b-it":             "google/gemma-2-9b-it",
    "gemma-2-27b-it":            "google/gemma-2-27b-it",
    # Qwen 1.5 (skip 72B)
    "qwen1.5-0.5b-chat":         "Qwen/Qwen1.5-0.5B-Chat",
    "qwen1.5-1.8b-chat":         "Qwen/Qwen1.5-1.8B-Chat",
    "qwen1.5-4b-chat":           "Qwen/Qwen1.5-4B-Chat",
    "qwen1.5-7b-chat":           "Qwen/Qwen1.5-7B-Chat",
    "qwen1.5-14b-chat":          "Qwen/Qwen1.5-14B-Chat",
    "qwen1.5-32b-chat":          "Qwen/Qwen1.5-32B-Chat",
    # Mistral / Mixtral (skip mistral-large 123B)
    "mistral-7b-instruct-v0.1":  "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral-7b-instruct-v0.2":  "mistralai/Mistral-7B-Instruct-v0.2",
    # Vicuna
    "vicuna-7b-v1.5":            "lmsys/vicuna-7b-v1.5",
    "vicuna-13b-v1.5":           "lmsys/vicuna-13b-v1.5",
    "vicuna-33b-v1.3":           "lmsys/vicuna-33b-v1.3",
    # Yi
    "yi-6b-chat":                "01-ai/Yi-6B-Chat",
    "yi-34b-chat":               "01-ai/Yi-34B-Chat",
    # Zephyr
    "zephyr-7b-beta":            "HuggingFaceH4/zephyr-7b-beta",
    # Dolphin
    "dolphin-2.2.1-mistral-7b":  "cognitivecomputations/dolphin-2.2.1-mistral-7b",
    "dolphin-2.6-mixtral-8x7b":  "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
    # Others
    "chatglm3-6b":               "THUDM/chatglm3-6b",
    "openchat-3.5-0106":         "openchat/openchat-3.5-0106",
    "aurora-m":                  "aurora-m/aurora-m-biden-harris-redteamed",
}


@dataclass
class Status:
    name: str
    repo: str
    state: str  # OK | PARTIAL | MISSING
    detail: str = ""


def inspect(name: str, repo: str) -> Status:
    path = OUTPUT_DIR / name
    if not path.is_dir():
        return Status(name, repo, "MISSING", "directory not found")
    files = set(os.listdir(path))
    idx_file = next((f for f in files if f.endswith(".index.json")), None)
    if idx_file:
        try:
            idx = json.loads((path / idx_file).read_text())
            needed = set(idx.get("weight_map", {}).values())
            missing = needed - files
            if missing:
                return Status(name, repo, "PARTIAL",
                              f"{len(missing)}/{len(needed)} shards missing")
            return Status(name, repo, "OK", f"{len(needed)} shards")
        except Exception as e:  # noqa: BLE001
            return Status(name, repo, "PARTIAL", f"bad index.json: {e}")
    has_weights = any(
        f.endswith(".safetensors") or (f.endswith(".bin") and "pytorch" in f)
        for f in files
    )
    if has_weights:
        return Status(name, repo, "OK", "single-file weights")
    return Status(name, repo, "MISSING", "no weight files")


def cmd_check(args: argparse.Namespace) -> int:
    rows = [inspect(n, r) for n, r in sorted(MODELS.items())]
    by_state = {"OK": [], "PARTIAL": [], "MISSING": []}
    for row in rows:
        by_state[row.state].append(row)
    for state in ("OK", "PARTIAL", "MISSING"):
        bucket = by_state[state]
        print(f"=== {state} ({len(bucket)}) ===")
        for r in bucket:
            print(f"  {r.name:32s}  {r.detail}" if r.detail else f"  {r.name}")
        print()
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfFolder, snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1

    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)
    token = HfFolder.get_token()
    if not token:
        print("⚠️  No HuggingFace token found. Gated models (Llama/Gemma/Mistral) may fail.")
        print("   Login via: huggingface-cli login")

    rows = [inspect(n, r) for n, r in sorted(MODELS.items())]

    if args.only:
        rows = [r for r in rows if r.name in set(args.only)]
        unknown = set(args.only) - {r.name for r in rows}
        if unknown:
            print(f"Unknown models: {sorted(unknown)}")
            return 2
    elif not args.include_ok:
        rows = [r for r in rows if r.state != "OK"]

    if not rows:
        print("Nothing to download (all models are complete).")
        return 0

    print(f"Will download/repair {len(rows)} models via {HF_MIRROR}")
    for r in rows:
        print(f"  - {r.name:32s}  [{r.state}] {r.detail}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []
    succeeded: list[str] = []

    for r in rows:
        local_path = OUTPUT_DIR / r.name
        print(f"--- Downloading {r.repo} -> {local_path} ---")
        try:
            snapshot_download(
                repo_id=r.repo,
                local_dir=str(local_path),
                resume_download=True,
                force_download=False,
                local_dir_use_symlinks=False,
                max_workers=args.workers,
                token=token,
            )
            succeeded.append(r.name)
            print(f"✅ {r.name}")
        except Exception as e:  # noqa: BLE001
            failed.append((r.name, str(e).splitlines()[0][:200]))
            print(f"❌ {r.name}: {failed[-1][1]}")
        print()

    print("=" * 60)
    print(f"Succeeded: {len(succeeded)}")
    for n in succeeded:
        print(f"  ✅ {n}")
    print(f"Failed: {len(failed)}")
    for n, err in failed:
        print(f"  ❌ {n}: {err}")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="Report which models are OK / PARTIAL / MISSING")

    dl = sub.add_parser("download", help="Download missing or incomplete models")
    dl.add_argument("--only", nargs="+", help="Only operate on these local model names")
    dl.add_argument("--include-ok", action="store_true",
                    help="Also re-fetch already-complete models (verify)")
    dl.add_argument("--workers", type=int, default=8, help="Parallel download workers")

    args = parser.parse_args()
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "download":
        return cmd_download(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
