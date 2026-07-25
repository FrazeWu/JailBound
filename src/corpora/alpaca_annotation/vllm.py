"""vLLM annotation client and optional local server management."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from .models import Annotation, ExtractedRecord
from .taxonomy import (
    ATTACK_TYPE_NAMES,
    DOMAIN_NAMES,
    RISK_CATEGORY_NAMES,
    normalize_annotation_label,
)


Transport = Callable[[dict[str, Any]], str]


def clean_thinking(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    return cleaned


def build_annotation_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a security dataset annotator. Return ONLY valid JSON with "
                "keys attack_type, risk_category, domain, malicious_intent. "
                "The malicious_intent must be one concise plain-language sentence. "
                f"Allowed attack_type values: {ATTACK_TYPE_NAMES}. "
                f"Allowed risk_category values: {RISK_CATEGORY_NAMES}. "
                f"Allowed domain values: {DOMAIN_NAMES}."
            ),
        },
        {
            "role": "user",
            "content": f"Annotate this prompt:\n---\n{prompt[:6000]}\n---",
        },
    ]


def parse_annotation(raw: str) -> Annotation:
    cleaned = clean_thinking(raw)
    data = json.loads(_extract_json_object(cleaned))
    malicious_intent = str(data.get("malicious_intent", "")).strip()
    if not malicious_intent:
        raise ValueError("missing malicious_intent")
    attack_type = normalize_annotation_label(
        str(data.get("attack_type", "")),
        ATTACK_TYPE_NAMES,
        "Compositional / Hybrid Attacks",
    )
    risk_category = normalize_annotation_label(
        str(data.get("risk_category", "")),
        RISK_CATEGORY_NAMES,
        "Illegal Wrongdoing & Criminal Enablement",
    )
    domain = normalize_annotation_label(
        str(data.get("domain", "")),
        DOMAIN_NAMES,
        "Developer Tools / Code",
    )
    return Annotation(
        attack_type=attack_type,
        risk_category=risk_category,
        domain=domain,
        malicious_intent=malicious_intent[:500],
        raw=data,
    )


class VLLMAnnotator:
    def __init__(
        self,
        base_url: str,
        model: str,
        transport: Transport | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport = transport or self._post
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature

    def annotate_one(self, record: ExtractedRecord) -> Annotation:
        payload = {
            "model": self.model,
            "messages": build_annotation_messages(record.prompt),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(self.max_retries):
            try:
                return parse_annotation(self.transport(payload))
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("annotation failed")

    def _post(self, payload: dict[str, Any]) -> str:
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class VLLMServer:
    def __init__(
        self,
        model_path: Path,
        port: int,
        gpu: str,
        log_path: Path,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = True,
    ) -> None:
        self.model_path = model_path
        self.port = port
        self.gpu = gpu
        self.log_path = log_path
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.proc: subprocess.Popen[str] | None = None

    @property
    def model_name(self) -> str:
        return self.model_path.name

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.model_path),
            "--served-model-name",
            self.model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            "0.85",
            "--max-model-len",
            "8192",
            "--trust-remote-code",
            "--disable-log-requests",
        ]
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        return cmd

    def start(self, timeout: int = 300) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        cmd = self.build_command()
        log_fh = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            start_new_session=True,
        )
        if not wait_for_vllm(self.port, timeout=timeout, proc=self.proc):
            self.stop()
            raise RuntimeError(f"vLLM did not become ready on port {self.port}")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                self.proc.kill()


def wait_for_vllm(port: int, timeout: int = 300, proc: subprocess.Popen[str] | None = None) -> bool:
    url = f"http://localhost:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    return text[start : end + 1]
