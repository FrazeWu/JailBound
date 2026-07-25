#!/usr/bin/env python3
"""
Dataset Download Script
Downloads datasets from HuggingFace, GitHub, and other sources based on CSV configuration.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import List
import logging
import shutil
import urllib.request
import tarfile
from zipfile import ZipFile

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.absolute()
DOWNLOAD_DIR = BASE_DIR / "corpora" / "downloaded_datasets"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_DIR = DOWNLOAD_DIR / "prompt"
BENCHMARK_DIR = DOWNLOAD_DIR / "benchmark"
PROMPT_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python3"

LOG_FILE = DOWNLOAD_DIR / "download.log"
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)


class DatasetDownloader:
    """Downloads datasets from various sources."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.success = []
        self.failed = []
        self.prompt_datasets = set()
        self.benchmark_datasets = set()

    @staticmethod
    def normalize_github_url(url: str) -> str:
        """
        Normalize GitHub URLs by removing tree/branch and subdirectory paths.

        Args:
            url: GitHub URL (may include /tree/main/, /tree/master/, subdirectories)

        Returns:
            Normalized repository URL
        """
        url = url.strip()
        if not url.startswith("https://github.com/"):
            return url

        parts = url.split("https://github.com/")
        if len(parts) != 2:
            return url

        repo_and_rest = parts[1]
        repo_parts = repo_and_rest.split("/")

        if len(repo_parts) < 2:
            return url

        repo_path = f"{repo_parts[0]}/{repo_parts[1]}"
        return f"https://github.com/{repo_path}"

    def download_huggingface(self, dataset_name: str, hf_url: str) -> bool:
        """
        Download dataset from HuggingFace using huggingface_hub Python API.

        Args:
            dataset_name: Name to save the dataset as
            hf_url: HuggingFace dataset URL

        Returns:
            True if successful, False otherwise
        """
        dataset_path = hf_url.replace("https://huggingface.co/datasets/", "")
        output_dir = DOWNLOAD_DIR / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {dataset_name} from HuggingFace: {dataset_path}")

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=dataset_path,
                repo_type="dataset",
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                tqdm_class=None,
                resume_download=True,
            )

            logger.info(f"✓ Successfully downloaded {dataset_name} via huggingface_hub")
            return True

        except Exception as e:
            logger.warning(f"huggingface_hub download failed: {e}, trying git clone")

            try:
                git_url = f"https://huggingface.co/datasets/{dataset_path}"

                env = os.environ.copy()
                env["GIT_TERMINAL_PROMPT"] = "0"

                cmd = ["git", "clone", "--depth", "1", git_url, str(output_dir)]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600, env=env
                )

                if result.returncode == 0:
                    logger.info(f"✓ Successfully cloned {dataset_name} via git")
                    return True
                else:
                    logger.warning(f"Git clone failed: {result.stderr[:200]}")

            except Exception as git_error:
                logger.warning(f"Git clone failed: {git_error}")

            logger.error(f"✗ Failed to download {dataset_name} from HuggingFace")
            return False

    def download_github(self, dataset_name: str, github_url: str) -> bool:
        """
        Download dataset from GitHub using git clone.

        Args:
            dataset_name: Name to save the dataset as
            github_url: GitHub URL

        Returns:
            True if successful, False otherwise
        """
        normalized_url = self.normalize_github_url(github_url)
        output_dir = DOWNLOAD_DIR / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cloning {dataset_name} from GitHub: {normalized_url}")

        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"

        result = subprocess.run(
            ["git", "clone", "--depth", "1", normalized_url, str(output_dir)],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        if result.returncode == 0:
            self._maybe_move_to_category(dataset_name, output_dir)
            logger.info(f"✓ Successfully cloned {dataset_name}")
            return True
        else:
            logger.error(f"✗ Failed to clone {normalized_url}: {result.stderr[:300]}")
            return False

    def download_other(self, dataset_name: str, url: str) -> bool:
        """
        Download dataset from other sources (HTTP, etc.).

        Args:
            dataset_name: Name to save the dataset as
            url: URL to download from

        Returns:
            True if successful, False otherwise
        """
        try:
            output_dir = DOWNLOAD_DIR / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {dataset_name} from: {url}")

            filename = url.split("/")[-1] or f"{dataset_name}.download"
            filepath = output_dir / filename

            def report_hook(block_num, block_size, total_size):
                percent = (
                    (block_num * block_size / total_size) * 100 if total_size > 0 else 0
                )
                sys.stdout.write(f"\rDownloading: {percent:.1f}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(url, filepath, reporthook=report_hook)
            print()

            if filepath.suffix == ".zip":
                logger.info(f"Extracting zip file: {filepath}")
                with ZipFile(filepath, "r") as zip_ref:
                    zip_ref.extractall(output_dir)
                filepath.unlink()
            elif filepath.suffix in [".tar", ".gz", ".tgz", ".bz2"]:
                logger.info(f"Extracting tar file: {filepath}")
                with tarfile.open(filepath, "r:*") as tar_ref:
                    tar_ref.extractall(output_dir)
                filepath.unlink()

            self._maybe_move_to_category(dataset_name, output_dir)
            logger.info(f"✓ Successfully downloaded {dataset_name}")
            return True

        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return False

    def download_dataset(self, dataset_name: str, sources: dict) -> bool:
        """
        Attempt to download a dataset from available sources.

        Args:
            dataset_name: Name of the dataset
            sources: Dictionary with 'download_links', 'github', 'huggingface', 'other'

        Returns:
            True if successfully downloaded, False otherwise
        """
        evaluation_type = sources.get("evaluation_type", "").strip().lower()
        download_links = sources.get("download_links", "").split(";")
        download_links = [link.strip() for link in download_links if link.strip()]

        hf_link = sources.get("huggingface", "").strip()
        if hf_link:
            hf_links = hf_link.split(";")
            for hf_url in hf_links:
                hf_url = hf_url.strip()
                if hf_url and huggingface_available():
                    logger.info(f"Trying HuggingFace: {hf_url}")
                    if self.download_huggingface(dataset_name, hf_url):
                        self._record_category(dataset_name, evaluation_type)
                        self.success.append(dataset_name)
                        return True

        github_link = sources.get("github", "").strip()
        if github_link:
            github_links = github_link.split(";")
            for github_url in github_links:
                github_url = github_url.strip()
                if github_url and git_available():
                    logger.info(f"Trying GitHub: {github_url}")
                    if self.download_github(dataset_name, github_url):
                        self._record_category(dataset_name, evaluation_type)
                        self.success.append(dataset_name)
                        return True

        other_link = sources.get("other", "").strip()
        if other_link:
            other_links = other_link.split(";")
            for other_url in other_links:
                other_url = other_url.strip()
                if other_url:
                    logger.info(f"Trying other source: {other_url}")
                    if self.download_other(dataset_name, other_url):
                        self._record_category(dataset_name, evaluation_type)
                        self.success.append(dataset_name)
                        return True

        for link in download_links:
            if "huggingface.co" in link and huggingface_available():
                if self.download_huggingface(dataset_name, link):
                    self._record_category(dataset_name, evaluation_type)
                    self.success.append(dataset_name)
                    return True
            elif "github.com" in link and git_available():
                if self.download_github(dataset_name, link):
                    self._record_category(dataset_name, evaluation_type)
                    self.success.append(dataset_name)
                    return True
            else:
                if self.download_other(dataset_name, link):
                    self._record_category(dataset_name, evaluation_type)
                    self.success.append(dataset_name)
                    return True

        logger.warning(f"✗ Failed to download {dataset_name} from all sources")
        self.failed.append((dataset_name, sources))
        return False


def git_available() -> bool:
    """Check if git is available."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def huggingface_available() -> bool:
    """Check if huggingface-cli is available."""
    try:
        subprocess.run(
            ["huggingface-cli", "--version"], capture_output=True, check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def load_csv(csv_path: Path) -> List[dict]:
    """Load datasets from CSV file."""
    datasets = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            datasets.append(row)

    return datasets


def main():
    """Main function to download all datasets."""
    csv_path = BASE_DIR / "corpora" / "safetyprompts_dataset_links.csv"

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    logger.info(f"Loading datasets from: {csv_path}")
    datasets = load_csv(csv_path)
    logger.info(f"Found {len(datasets)} datasets to download")

    # Check available tools
    logger.info(f"Git available: {git_available()}")
    logger.info(f"HuggingFace CLI available: {huggingface_available()}")

    downloader = DatasetDownloader(BASE_DIR)

    # Download each dataset
    for i, dataset in enumerate(datasets, 1):
        dataset_name = dataset.get("dataset", "").strip()
        if not dataset_name:
            continue

        sources = {
            "download_links": dataset.get("download_links", ""),
            "github": dataset.get("github", ""),
            "huggingface": dataset.get("huggingface", ""),
            "other": dataset.get("other", ""),
        }

        logger.info(f"\n[{i}/{len(datasets)}] Processing: {dataset_name}")
        downloader.download_dataset(dataset_name, sources)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successfully downloaded: {len(downloader.success)}")
    logger.info(f"Failed: {len(downloader.failed)}")

    if downloader.failed:
        logger.info("\nFailed datasets:")
        for dataset_name, sources in downloader.failed:
            logger.info(f"  - {dataset_name}")

    logger.info(f"\nDatasets saved to: {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
