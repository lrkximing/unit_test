from functools import lru_cache
import os
import time
from typing import Dict, Optional

from langchain_core.prompts import PromptTemplate
import openai
import transformers
import yaml


DEFAULT_BASE_URL = "https://aigc.x-see.cn/v1"
DEFAULT_MODEL_NAME = "deepseek-v3.2"


def load_local_model(model_path):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    return model, tokenizer


def local_model_generation(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _resolve_config_value(config: Optional[Dict], field: str, env_field: str, fallback: Optional[str] = None) -> Optional[str]:
    config = config or {}
    value = config.get(field)
    if value:
        return value

    env_name = config.get(env_field)
    if env_name:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value

    return fallback


@lru_cache(maxsize=8)
def _get_openai_client(api_key: str, base_url: str):
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def gpt_api(prompt, temperature=0.0, model_name=None, api_config=None):
    config = dict(api_config or {})
    model = model_name or config.get("model") or os.getenv("LLM_MODEL") or DEFAULT_MODEL_NAME
    has_custom_api_key = bool(config.get("api_key") or config.get("api_key_env"))
    has_custom_base_url = bool(config.get("base_url") or config.get("base_url_env"))
    api_key = _resolve_config_value(
        config,
        "api_key",
        "api_key_env",
        fallback=None if has_custom_api_key else os.getenv("LLM_API_KEY"),
    )
    base_url = _resolve_config_value(
        config,
        "base_url",
        "base_url_env",
        fallback=None if has_custom_base_url else (os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL),
    )

    if not api_key:
        raise ValueError(f"模型 {model} 缺少 API key 配置。")
    if not base_url:
        raise ValueError(f"模型 {model} 缺少 base_url 配置。")

    client = _get_openai_client(api_key, base_url)
    max_attempts = max(1, int(os.getenv("RACA_LLM_RETRIES", "3") or "3"))
    request_timeout = float(os.getenv("RACA_LLM_TIMEOUT", "180") or "180")
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=False,
                timeout=request_timeout,
            )
            break
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            message = str(exc)
            permanent_error = (
                "InvalidParameter" in message
                or "invalid_parameter" in message
                or "Range of input length" in message
                or status_code in {400, 401, 403, 404}
            )
            retryable = (
                not permanent_error
                and (
                    status_code in {408, 429, 500, 502, 503, 504}
                    or exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError", "TimeoutException"}
                    or "Bad Gateway" in message
                )
            )
            if attempt >= max_attempts or not retryable:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    else:
        raise last_error  # defensive; loop either breaks or raises
    message = response.choices[0].message.content if response.choices else ""
    return message or ""


def load_prompt_from_yaml(config_path, key):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get(key, "")
