#!/usr/bin/env python3
"""
Redownload failed datasets with corrected URLs and HuggingFace sources.
"""

import csv
import os
import subprocess
import sys
import shutil
from pathlib import Path
import logging

BASE_DIR = Path(__file__).parent.parent.absolute()

from download_datasets import DatasetDownloader, DOWNLOAD_DIR, git_available

# Dataset-specific URL corrections and alternative sources
RETRY_DATASETS = {
    "AART": {
        "sources": ["huggingface.co/datasets/davanstrien/aart-ai-safety-dataset"],
        "type": "hf",
    },
    "WinoBias": {"sources": ["huggingface.co/datasets/wino_bias"], "type": "hf"},
    "SafetyInstructions": {
        "sources": ["github.com/vinid/safety-tuned-llamas"],
        "type": "github",
        "subdir": "data/training",
    },
    "StrongREJECT": {
        "sources": ["github.com/alexandrasouly/strongreject"],
        "type": "github",
        "force_retries": True,
    },
    "LMBias": {
        "sources": ["github.com/pliang279/LM_bias"],
        "type": "github",
        "subdir": "data",
    },
    "LatentJailbreak": {
        "sources": ["github.com/qiuhuachuan/latent-jailbreak"],
        "type": "github",
        "force_retries": True,
    },
    # Other datasets might need manual investigation
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_huggingface_manual(dataset_name: str, hf_path: str) -> bool:
    """Download from HuggingFace using Python API."""
    output_dir = DOWNLOAD_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset_name} from HuggingFace: {hf_path}")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_path,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            tqdm_class=None,
            resume_download=True,
        )

        logger.info(f"✓ Successfully downloaded {dataset_name}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download {dataset_name}: {e}")
        return False


def download_github_with_retries(
    dataset_name: str, github_url: str, force_retries: int = 3
) -> bool:
    """Download from GitHub with retry logic for problematic repos."""
    output_dir = DOWNLOAD_DIR / dataset_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize URL - handle various formats like:
    # - "github.com/user/repo"
    # - "user/repo"
    # - "https://github.com/user/repo/tree/main/subdir"
    if github_url.startswith("https://github.com/"):
        if "/tree/" in github_url:
            parts = github_url.split("github.com/")
            repo_path = parts[1].split("/tree/")[0]
            github_url = f"https://github.com/{repo_path}"
    elif github_url.startswith("github.com/"):
        repo_path = github_url.split("github.com/")[1]
        if "/tree/" in repo_path:
            repo_path = repo_path.split("/tree/")[0]
        github_url = f"https://github.com/{repo_path}"
    else:
        if "/tree/" in github_url:
            github_url = github_url.split("/tree/")[0]
        github_url = f"https://github.com/{github_url}"

    logger.info(f"Cloning {dataset_name} from GitHub: {github_url}")

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    for attempt in range(force_retries):
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, str(output_dir)],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        if result.returncode == 0:
            logger.info(f"✓ Successfully cloned {dataset_name}")
            return True
        else:
            logger.warning(
                f"Attempt {attempt + 1}/{force_retries} failed: {result.stderr[:200]}"
            )

    logger.error(f"✗ Failed to clone {dataset_name} after {force_retries} attempts")
    return False


def main():
    logger.info(f"Starting redownload of failed datasets")

    # Check available tools
    hf_available = False
    try:
        from huggingface_hub import snapshot_download

        hf_available = True
        logger.info("HuggingFace Python API available")
    except ImportError:
        logger.warning("HuggingFace Python API not available")

    git_available_here = git_available()
    logger.info(f"Git available: {git_available_here}")

    success = []
    failed = []

    for dataset_name, config in RETRY_DATASETS.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"{'=' * 60}")

        output_dir = DOWNLOAD_DIR / dataset_name
        if output_dir.exists():
            logger.info(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)

        # Try HuggingFace first
        if config["type"] == "hf" and hf_available:
            for source in config["sources"]:
                hf_path = source.replace("huggingface.co/datasets/", "")
                if download_huggingface_manual(dataset_name, hf_path):
                    success.append(dataset_name)
                    break

        # Try GitHub
        if config["type"] == "github" and git_available_here:
            for source in config["sources"]:
                retries = config.get("force_retries", 1)
                if download_github_with_retries(
                    dataset_name, source, force_retries=retries
                ):
                    success.append(dataset_name)
                    break

        if dataset_name not in success:
            failed.append(dataset_name)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("REDOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successfully downloaded: {len(success)}")
    logger.info(f"Failed: {len(failed)}")

    if success:
        logger.info("\nSuccessful downloads:")
        for name in success:
            logger.info(f"  ✓ {name}")

    if failed:
        logger.info("\nFailed downloads:")
        for name in failed:
            logger.info(f"  ✗ {name}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
