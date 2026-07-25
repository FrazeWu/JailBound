"""
vllm_client.py — vLLM OpenAI-compatible API 调用封装
支持批量并发请求和重试机制
"""

import os
import requests
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.environ.get("BENCHMARK_API_BASE_URL", "http://localhost:8000")
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def get_available_models():
    """查询 vLLM 服务上可用的模型列表"""
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m["id"] for m in data.get("data", [])]
        return models
    except Exception as e:
        logger.error(f"Failed to query models: {e}")
        return []


def chat_completion(
    messages,
    model=None,
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    **kwargs,
):
    """
    调用 vLLM 的 OpenAI-compatible /v1/chat/completions 接口

    Args:
        messages: OpenAI 格式的消息列表 [{"role": "...", "content": "..."}]
        model: 模型名称 (None 则自动获取第一个可用模型)
        max_tokens: 最大生成 token 数
        temperature: 采样温度

    Returns:
        str: 模型生成的文本内容
    """
    if model is None:
        models = get_available_models()
        if not models:
            raise RuntimeError("No models available on vLLM server")
        model = models[0]
        logger.info(f"Auto-selected model: {model}")

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY * (attempt + 1))
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def batch_chat_completion(
    messages_list,
    model=None,
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    max_workers=8,
    progress_callback=None,
    **kwargs,
):
    """
    批量并发调用 chat_completion

    Args:
        messages_list: 消息列表的列表 [[msg1, msg2], [msg1, msg2], ...]
        model: 模型名称
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        max_workers: 最大并发数
        progress_callback: 进度回调 fn(completed, total)

    Returns:
        list: 按输入顺序返回的结果列表，失败项为 None
    """
    if model is None:
        models = get_available_models()
        if not models:
            raise RuntimeError("No models available on vLLM server")
        model = models[0]
        logger.info(f"Auto-selected model: {model}")

    total = len(messages_list)
    results = [None] * total
    completed = 0

    def _call(idx, messages):
        return idx, chat_completion(messages, model=model, max_tokens=max_tokens, temperature=temperature, **kwargs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_call, i, msgs): i
            for i, msgs in enumerate(messages_list)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, result = future.result()
                results[idx] = result
            except Exception as e:
                logger.error(f"Failed for index {idx}: {e}")
                results[idx] = None

            completed += 1
            if progress_callback and completed % 100 == 0:
                progress_callback(completed, total)

    return results
