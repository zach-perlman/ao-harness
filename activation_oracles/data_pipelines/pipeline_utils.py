"""
Shared utilities for dataset generation pipelines.

Provides:
- model_dir_name: Convert HF model ID to a directory-safe short name
- async_api_call: Anthropic API calls with retry + high-priority key fallback
- parse_json_response: Extract JSON from LLM responses (handles fences, trailing commas, etc.)
- extract_tool_input: Extract structured output from tool_use responses
- run_concurrent: Run async tasks with a concurrency limit and progress tracking
"""

import argparse
import asyncio
import functools
import json
import os
import re
from pathlib import Path

import anthropic
from anthropic._exceptions import APIStatusError, APIConnectionError, InternalServerError, OverloadedError, RateLimitError

# Force unbuffered output so we can monitor progress in background
print = functools.partial(print, flush=True)


def vllm_gpu_util(default: float) -> float:
    """vLLM gpu_memory_utilization, overridable via AO_VLLM_GPU_UTIL.

    Lets a dataset generator share the GPU with the resident local judge
    (e.g. generator 0.5 + judge 0.3) instead of OOMing at vLLM init.
    """
    return float(os.environ.get("AO_VLLM_GPU_UTIL", default))


def model_dir_name(model_id: str) -> str:
    """Convert a HuggingFace model ID to a directory-safe short name.

    Examples:
        "Qwen/Qwen3-8B" -> "Qwen3-8B"
        "meta-llama/Llama-3-8B" -> "Llama-3-8B"
        "Qwen3-8B" -> "Qwen3-8B"
    """
    return model_id.split("/")[-1]


def add_model_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add a required --model argument to an argparse parser."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen3-8B)",
    )
    return parser


def add_n_per_task_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add an optional --n-per-task argument (target eval-set size for this task).

    None ⇒ keep the script's built-in default. Generators that are bounded by a
    fixed hand-written source (e.g. missing_info's PROBLEMS list) accept the flag
    for a uniform CLI but cannot exceed that bound; they document the no-op.
    """
    parser.add_argument(
        "--n-per-task",
        type=int,
        default=None,
        help="target number of eval items for this task (None = script default)",
    )
    return parser


def load_dotenv():
    """Load .env from repo root if API keys aren't already set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    # Walk up from this file to find .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        if key and value:
            # Strip inline comments (e.g. KEY=value # comment)
            if " #" in value:
                value = value[:value.index(" #")]
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

# In local-judge mode the Anthropic client is constructed by callers but never
# used (async_api_call reroutes). Give it a placeholder key so construction
# doesn't raise when no real ANTHROPIC_API_KEY is present.
if os.environ.get("AO_JUDGE_BASE_URL") and not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = "local-judge-placeholder"

# ---------------------------------------------------------------------------
# Local judge backend (OpenAI-compatible, e.g. vLLM on 127.0.0.1:8001/v1).
#
# When AO_JUDGE_BASE_URL is set, every judge call funnelled through
# async_api_call is transparently rerouted to that endpoint instead of
# Anthropic. We translate Anthropic's request shape (system + messages +
# thinking + tools/tool_choice) to OpenAI chat.completions and wrap the reply
# in a thin shim so existing callers (`resp.content[0].text`,
# `extract_tool_input(resp)`) work unchanged.
# ---------------------------------------------------------------------------

def _judge_base_url() -> str | None:
    return os.environ.get("AO_JUDGE_BASE_URL")


def _local_judge_model() -> str:
    return os.environ.get("AO_JUDGE_MODEL", "local-judge")


class _Text:
    """Mimics an Anthropic text content block."""
    type = "text"
    def __init__(self, text: str):
        self.text = text


class _ToolUse:
    """Mimics an Anthropic tool_use content block."""
    type = "tool_use"
    def __init__(self, name: str, tool_input: dict):
        self.name = name
        self.input = tool_input


class _LocalResponse:
    """Anthropic-shaped wrapper around an OpenAI chat.completions message."""
    def __init__(self, message):
        blocks: list = []
        for call in getattr(message, "tool_calls", None) or []:
            args = call.function.arguments
            blocks.append(_ToolUse(call.function.name, json.loads(args) if isinstance(args, str) else args))
        text = message.content or ""
        # Strip a leading <think>...</think> reasoning block if present.
        text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        blocks.append(_Text(text))
        self.content = blocks


_LOCAL_CLIENT = None


def _get_local_client():
    global _LOCAL_CLIENT
    if _LOCAL_CLIENT is None:
        from openai import AsyncOpenAI
        _LOCAL_CLIENT = AsyncOpenAI(base_url=_judge_base_url(), api_key="unused")
    return _LOCAL_CLIENT


def _to_openai_kwargs(kwargs: dict) -> dict:
    """Translate Anthropic messages.create kwargs to OpenAI chat.completions."""
    messages = list(kwargs.get("messages", []))
    if kwargs.get("system"):
        messages = [{"role": "system", "content": kwargs["system"]}] + messages
    out: dict = {
        "model": _local_judge_model(),
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1024),
        "temperature": kwargs.get("temperature", 1.0),
        # Disable the model's <think> phase at the source (vLLM extension) so
        # the judge answers directly; `thinking` (Anthropic-only) is dropped.
        "extra_body": {"chat_template_kwargs": {
            "enable_thinking": os.environ.get("AO_JUDGE_ENABLE_THINKING", "0") == "1"}},
    }
    if "tools" in kwargs:  # Anthropic tool schema -> OpenAI function schema
        out["tools"] = [
            {"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t["input_schema"]}}
            for t in kwargs["tools"]
        ]
    tc = kwargs.get("tool_choice")
    if isinstance(tc, dict) and tc.get("type") == "tool":
        out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
    return out


async def async_api_call(client, *, max_retries: int = 10, **kwargs):
    """Make a judge API call with retry and backoff.

    Routes to a local OpenAI-compatible server when AO_JUDGE_BASE_URL is set
    (the passed Anthropic `client` is ignored in that mode); otherwise calls
    Anthropic. Retries transient errors with linear backoff. Does NOT manage
    concurrency — wrap calls in a semaphore at the task level.
    """
    if _judge_base_url():
        local = _get_local_client()
        oai_kwargs = _to_openai_kwargs(kwargs)
        for attempt in range(max_retries):
            try:
                resp = await local.chat.completions.create(**oai_kwargs)
                return _LocalResponse(resp.choices[0].message)
            except Exception as e:  # local server transient errors
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"    [local judge retry {attempt+1}/{max_retries} → {wait}s: {type(e).__name__}]")
                    await asyncio.sleep(wait)
                else:
                    raise
    for attempt in range(max_retries):
        try:
            return await client.messages.create(**kwargs)
        except (OverloadedError, RateLimitError, InternalServerError, APIConnectionError) as e:
            error_code = getattr(e, 'status_code', type(e).__name__)
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    [{error_code} retry {attempt+1}/{max_retries} → waiting {wait}s]")
                await asyncio.sleep(wait)
            else:
                raise


def parse_json_response(text: str):
    """Extract JSON from an LLM response, handling markdown fences and common issues."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Fix common LLM JSON issues
    cleaned = re.sub(r',\s*([}\]])', r'\1', text)  # trailing commas
    cleaned = re.sub(r':\s*\'([^\']*)\'\s*([,}\]])', r': "\1"\2', cleaned)  # single quotes to double
    # Fix unescaped backslashes (e.g. \boxed, \text) — escape any \ not followed by valid JSON escape chars
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Aggressive backslash escaping: replace ALL unescaped backslashes
    # (the regex above misses some edge cases with consecutive backslashes)
    aggressive = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu\\])', r'\\\\', cleaned)
    try:
        return json.loads(aggressive)
    except json.JSONDecodeError:
        pass
    # Scan for JSON array or object and use raw_decode to stop at the end
    decoder = json.JSONDecoder()
    for attempt_text in (aggressive, cleaned):
        for start_char in ("[", "{"):
            idx = attempt_text.find(start_char)
            if idx != -1:
                try:
                    obj, _ = decoder.raw_decode(attempt_text, idx)
                    return obj
                except json.JSONDecodeError:
                    continue
    raise json.JSONDecodeError("No JSON found", text, 0)


def extract_tool_input(resp) -> dict:
    """Extract tool use input from a response that used tool_choice.

    Use this when you force a tool call via tool_choice={"type": "tool", "name": "..."}.
    The response is guaranteed to be valid JSON (no parse_json_response needed).
    """
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("No tool_use block in response")


async def run_concurrent(tasks_fn, items, *, concurrency, label="", progress_interval=10):
    """Run async tasks with a concurrency limit and progress tracking.

    Args:
        tasks_fn: async function(semaphore, i, item) -> None. Must handle its own errors.
        items: list of items to process
        concurrency: max concurrent tasks
        label: name for progress prints
        progress_interval: print progress every N completions
    """
    semaphore = asyncio.Semaphore(concurrency)
    progress = {"done": 0}
    total = len(items)

    async def wrapped(i, item):
        async with semaphore:
            await tasks_fn(i, item)
        progress["done"] += 1
        d = progress["done"]
        if d % progress_interval == 0 or d == total or d <= 3:
            print(f"  {label} progress: {d}/{total}")

    await asyncio.gather(*[wrapped(i, item) for i, item in enumerate(items)])
