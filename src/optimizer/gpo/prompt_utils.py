"""The utility functions for prompting GPT and other models.

Lightweight by default: avoid importing heavy libraries until needed.
"""

import os
import time

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

model2model_path = {
    "llama2-chat-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-chat-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-chat-7b": "meta-llama/Llama-2-7b-hf",
}

_openai_client: OpenAI | None = None
_openai_client_kwargs: dict[str, str] = {}


def configure_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    reset: bool = True,
) -> None:
    """Configure the shared OpenAI client with explicit credentials.

    Passing an empty string will be ignored so env vars still apply. Clearing the
    configuration can be done by calling with both parameters omitted.
    """

    global _openai_client, _openai_client_kwargs

    if reset:
        _openai_client_kwargs.clear()

    if api_key is not None:
        api_key = api_key.strip()
        if api_key:
            _openai_client_kwargs["api_key"] = api_key
        else:
            _openai_client_kwargs.pop("api_key", None)

    if base_url is not None:
        base_url = base_url.strip()
        if base_url:
            _openai_client_kwargs["base_url"] = base_url
        else:
            _openai_client_kwargs.pop("base_url", None)

    _openai_client = None


def _get_openai_client() -> OpenAI:
    """Return a cached OpenAI client instance respecting env overrides."""

    global _openai_client

    if _openai_client is None:
        client_kwargs = dict(_openai_client_kwargs)

        if "api_key" not in client_kwargs:
            env_key = os.getenv("OPENAI_API_KEY") or os.getenv("QNAPI")
            if env_key:
                client_kwargs["api_key"] = env_key.strip()

        if "base_url" not in client_kwargs:
            env_base_url = (
                os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("QNAPIURL")
                or os.getenv("BASE_URL")
            )
            if env_base_url:
                client_kwargs["base_url"] = env_base_url.strip()

        _openai_client = OpenAI(**client_kwargs)

    return _openai_client


def call_openai_server_func(
    prompt, n=1, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
    """The function to call OpenAI server with an input string."""
    try:
        client = _get_openai_client()

        if isinstance(prompt, str):
            completion = client.chat.completions.create(
                model=model,
                n=n,
                temperature=temperature,
                max_tokens=max_decode_steps,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            completions_list = []
            for i in range(n):
                completions_list.append(
                    completion.choices[i].message.content
                )
            # return completion.choices[0].message.content
            return completions_list
        elif isinstance(prompt, list):
            completion = client.chat.completions.create(
                model=model,
                n=n,
                temperature=temperature,
                max_tokens=max_decode_steps,
                messages=prompt,
            )
            completions_list = []
            for i in range(n):
                completions_list.append(
                    completion.choices[i].message.content
                )
            # return completion.choices[0].message.content
            return completions_list

    except APITimeoutError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except APIError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except APIConnectionError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"API connection error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except APIStatusError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"Service unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except AuthenticationError as e:
        print("OpenAI 认证失败，请确认 API Key 有效。详情:", getattr(e, "message", e))
        if hasattr(e, "body") and e.body:
            print("错误响应:", e.body)
        raise

    except PermissionDeniedError as e:
        print("OpenAI 权限错误，请检查账户权限。详情:", getattr(e, "message", e))
        if hasattr(e, "body") and e.body:
            print("错误响应:", e.body)
        raise

    except BadRequestError as e:
        print("OpenAI 返回 400 错误，请检查入参。详情:", getattr(e, "message", e))
        if hasattr(e, "body") and e.body:
            print("错误响应:", e.body)
        raise

    except OSError as e:
        retry_time = 5  # Adjust the retry time as needed
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

def call_vllm_server_func(
    prompt, model, llm_name, max_decode_steps=20, temperature=0.8, stop_tokens=None
):
    """The function to call vllm with a list of input strings.

    Imports vllm lazily to avoid heavy deps when not used.
    """

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_decode_steps, stop=stop_tokens
    )
    res_completions = []

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(prompt, list):
        if all(isinstance(elem, str) for elem in prompt):
            completions = model.generate(prompts=prompt, sampling_params=sampling_params)
        elif all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in prompt):
            completions = model.generate(prompt_token_ids=prompt, sampling_params=sampling_params)

    for output in completions:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        res_completions.append(generated_text)

    return res_completions


def load_vllm_llm(model, tensor_parallel_size):
    from vllm import LLM
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.35,  # 降低到35%显存占用（每卡~14GB）
        trust_remote_code=True,
        max_model_len=1024,  # 进一步限制序列长度到1024
        enable_prefix_caching=False,  # 禁用prefix caching节省显存
        max_num_batched_tokens=1024,  # 减少批处理token数到1024
        enforce_eager=True,  # 禁用torch.compile减少显存和初始化时间
        block_size=16,  # block_size必须是16的倍数（vLLM约束）
        swap_space=8,  # 增加swap空间使用CPU内存缓冲
    )
    return llm


if __name__ == "__main__":
    prompt = "The sun rises from the west."
    res_list = call_openai_server_func(prompt, max_decode_steps=50, n=8)
    print(res_list)
    print(res_list[0])
    print(type(res_list[0]))
