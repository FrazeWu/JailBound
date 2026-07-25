#!/usr/bin/env python3
"""Serial per-file HuggingFace downloader for large model repos."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi, HfFolder, hf_hub_download, hf_hub_url

from manage_models import FAMILY, MODELS, OUTPUT_DIR

ALWAYS_INCLUDE = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "tokenizer.json",
    "tekken.json",
    "params.json",
    "added_tokens.json",
]

INDEX_PRIORITY = [
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
]
REPO_FILES_CACHE_NAME = ".hf_repo_files.json"
HTTP_RETRIES = 3


@dataclass(frozen=True)
class DownloadEntry:
    filename: str
    transport: str  # hub | http


def load_index_payload(index_path: Path) -> dict:
    return json.loads(index_path.read_text(encoding="utf-8"))


def _unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def plan_download_filenames(
    repo_files: list[str],
    *,
    index_payload: dict | None = None,
) -> list[str]:
    repo_set = set(repo_files)
    plan: list[str] = [name for name in ALWAYS_INCLUDE if name in repo_set]

    index_name = next((name for name in INDEX_PRIORITY if name in repo_set), None)
    if index_name:
        plan.append(index_name)
        if index_payload is None:
            raise ValueError(f"index payload required for {index_name}")
        shard_names = sorted(set(index_payload.get("weight_map", {}).values()))
        plan.extend(name for name in shard_names if name in repo_set)
        return _unique_in_order(plan)

    weight_files = sorted(
        name
        for name in repo_files
        if name.endswith(".safetensors")
        or (name.endswith(".bin") and "pytorch_model" in name)
    )
    plan.extend(weight_files)
    return _unique_in_order(plan)


def plan_download_entries(
    repo_files: list[str],
    *,
    index_payload: dict | None = None,
) -> list[DownloadEntry]:
    filenames = plan_download_filenames(repo_files, index_payload=index_payload)
    entries: list[DownloadEntry] = []
    for filename in filenames:
        transport = "http" if (
            filename.endswith(".safetensors")
            or (filename.endswith(".bin") and "pytorch_model" in filename)
        ) else "hub"
        entries.append(DownloadEntry(filename=filename, transport=transport))
    return entries


def _local_model_dir(model_name: str) -> Path:
    return OUTPUT_DIR / FAMILY[model_name] / model_name


def _repo_files_cache_path(local_dir: Path) -> Path:
    return local_dir / REPO_FILES_CACHE_NAME


def load_repo_files_cache(local_dir: Path) -> list[str] | None:
    cache_path = _repo_files_cache_path(local_dir)
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"invalid repo files cache: {cache_path}")
    return data


def save_repo_files_cache(local_dir: Path, repo_files: list[str]) -> None:
    cache_path = _repo_files_cache_path(local_dir)
    cache_path.write_text(json.dumps(repo_files, indent=2, ensure_ascii=True), encoding="utf-8")


def fetch_repo_files(
    repo_id: str,
    *,
    local_dir: Path,
    token: str | None,
    endpoint: str,
    retries: int = 4,
    retry_delay: int = 5,
) -> list[str]:
    api = HfApi(endpoint=endpoint, token=token)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            info = api.model_info(repo_id)
            repo_files = [s.rfilename for s in (info.siblings or [])]
            save_repo_files_cache(local_dir, repo_files)
            return repo_files
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(f"RETRY model_info {attempt}/{retries}: {exc}", flush=True)
            if attempt < retries:
                time.sleep(retry_delay)

    cached = load_repo_files_cache(local_dir)
    if cached:
        print(f"FALLBACK repo file cache for {repo_id}", flush=True)
        return cached
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to fetch repo files for {repo_id}")


def _fetch_index_payload(
    repo_id: str,
    index_name: str,
    local_dir: Path,
    token: str | None,
    endpoint: str,
) -> dict:
    local_index = local_dir / index_name
    if local_index.exists():
        return load_index_payload(local_index)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=index_name,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
        endpoint=endpoint,
        etag_timeout=30,
    )
    return load_index_payload(Path(downloaded))


def build_download_plan(
    repo_id: str,
    local_dir: Path,
    token: str | None,
    endpoint: str,
) -> list[DownloadEntry]:
    local_dir.mkdir(parents=True, exist_ok=True)
    repo_files = fetch_repo_files(
        repo_id,
        local_dir=local_dir,
        token=token,
        endpoint=endpoint,
    )

    index_name = next((name for name in INDEX_PRIORITY if name in repo_files), None)
    index_payload = None
    if index_name:
        index_payload = _fetch_index_payload(repo_id, index_name, local_dir, token, endpoint)

    return plan_download_entries(repo_files, index_payload=index_payload)


def _wget_env() -> dict[str, str]:
    env = os.environ.copy()
    for upper, lower in (("HTTP_PROXY", "http_proxy"), ("HTTPS_PROXY", "https_proxy"), ("NO_PROXY", "no_proxy")):
        if env.get(upper) and not env.get(lower):
            env[lower] = env[upper]
    return env


def _http_download_file(
    repo_id: str,
    filename: str,
    destination: Path,
    token: str | None,
    endpoint: str,
) -> None:
    url = hf_hub_url(repo_id, filename, endpoint=endpoint)
    part_path = destination.with_name(destination.name + ".part")
    wget_cmd = [
        "wget",
        "--continue",
        f"--output-document={part_path}",
    ]
    curl_cmd = [
        "curl",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--output",
        str(part_path),
    ]
    if token:
        header = f"Authorization: Bearer {token}"
        wget_cmd.append(f"--header={header}")
        curl_cmd.extend(["--header", header])
    wget_cmd.append(url)
    curl_cmd.append(url)

    last_error: Exception | None = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            subprocess.run(wget_cmd, check=True, env=_wget_env())
            last_error = None
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"RETRY wget {filename} {attempt}/{HTTP_RETRIES}: {exc}", flush=True)
            if attempt < HTTP_RETRIES:
                time.sleep(2)
    if last_error is not None:
        for attempt in range(1, HTTP_RETRIES + 1):
            try:
                subprocess.run(curl_cmd, check=True, env=_wget_env())
                last_error = None
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
                print(f"RETRY curl {filename} {attempt}/{HTTP_RETRIES}: {exc}", flush=True)
                if attempt < HTTP_RETRIES:
                    time.sleep(5)
    if last_error is not None:
        raise last_error
    part_path.replace(destination)


def download_entry(
    repo_id: str,
    local_dir: Path,
    entry: DownloadEntry,
    token: str | None,
    endpoint: str,
) -> None:
    filename = entry.filename
    target = local_dir / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"SKIP {filename}", flush=True)
        return

    print(f"START {filename}", flush=True)
    if entry.transport == "hub":
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=token,
                endpoint=endpoint,
                etag_timeout=30,
            )
            print(f"DONE {path}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"FALLBACK HTTP {filename}: {exc}", flush=True)
            _http_download_file(repo_id, filename, target, token, endpoint)
            print(f"DONE {target}", flush=True)
            return

    _http_download_file(repo_id, filename, target, token, endpoint)
    print(f"DONE {target}", flush=True)


def download_plan(
    repo_id: str,
    local_dir: Path,
    entries: list[DownloadEntry],
    token: str | None,
    endpoint: str,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        download_entry(repo_id, local_dir, entry, token, endpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = HfFolder.get_token()
    repo_id = MODELS[args.model]
    local_dir = _local_model_dir(args.model)
    entries = build_download_plan(repo_id, local_dir, token, args.endpoint)

    print(f"MODEL {args.model}", flush=True)
    print(f"REPO {repo_id}", flush=True)
    print(f"FILES {len(entries)}", flush=True)
    for entry in entries:
        print(f"  {entry.filename} [{entry.transport}]", flush=True)

    if args.plan_only:
        return 0

    download_plan(repo_id, local_dir, entries, token, args.endpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
