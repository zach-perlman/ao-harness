"""Manage the local LLM judge (vLLM OpenAI-compatible server) as a supervisor service.

Gears-level overview:
  AOBench judging and dataset generation normally call Anthropic models; we
  replace them with a local Qwen3.5-35B-A3B (FP8 MoE) served by vLLM on
  127.0.0.1:<port>/v1. Per the instance conventions (CLAUDE.md §7) the server
  runs as a supervisor service, not a loose process:

    up     -> write wrapper script + supervisor conf (idempotent), start,
              then block until GET /v1/models answers (model load ~minutes).
    down   -> stop the service (frees its GPU memory slice).
    status -> supervisor state + live /v1/models probe.

  The service binds localhost only (never exposed through Caddy) and uses a
  vLLM reasoning parser so judge responses arrive in `content` without
  <think> blocks, keeping every downstream JSON-parsing judge happy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

from . import ENVS, load_config

SERVICE = "ao_judge"
WRAPPER = f"/opt/supervisor-scripts/{SERVICE}.sh"
SUPERVISOR_CONF = f"/etc/supervisor/conf.d/{SERVICE}.conf"


def _install(cfg: dict) -> None:
    j = cfg["judge"]
    vllm_bin = ENVS / "vllm" / "bin" / "vllm"
    extra = " ".join(j.get("extra_args", []))
    # Qwen3.5 emits tool calls and <think> blocks in its own formats, so we use
    # the model-matched vLLM parsers (the generic `hermes` parser mis-reads
    # Qwen's tool-call syntax). `--reasoning-parser qwen3` routes any reasoning
    # into `reasoning_content`, keeping `content` clean for the JSON-parsing
    # judges; the server-side default mirrors config's `enable_thinking`.
    chat_kwargs = json.dumps({"enable_thinking": bool(j.get("enable_thinking", False))})
    wrapper = f"""#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${{utils}}/logging.sh"
. "${{utils}}/environment.sh"

pty {vllm_bin} serve {j["model"]} \\
    --served-model-name {j["served_name"]} \\
    --host 127.0.0.1 --port {j["port"]} \\
    --gpu-memory-utilization {j["gpu_memory_utilization"]} \\
    --max-model-len {j["max_model_len"]} \\
    --max-num-seqs {j.get("max_num_seqs", 128)} \\
    --reasoning-parser qwen3 \\
    --default-chat-template-kwargs '{chat_kwargs}' \\
    --enable-auto-tool-choice --tool-call-parser qwen3_coder {extra} 2>&1
"""
    conf = f"""[program:{SERVICE}]
environment=PROC_NAME="%(program_name)s"
command={WRAPPER}
autostart=false
autorestart=unexpected
stopwaitsecs=60
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
"""
    for path, content in [(WRAPPER, wrapper), (SUPERVISOR_CONF, conf)]:
        with open(path, "w") as f:
            f.write(content)
    subprocess.run(["chmod", "+x", WRAPPER], check=True)
    subprocess.run(["supervisorctl", "reread"], check=True, capture_output=True)
    subprocess.run(["supervisorctl", "update"], check=True, capture_output=True)


def _probe(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def up(cfg: dict, timeout_s: int = 2400) -> None:
    # Cold start of the 35B-A3B FP8 hybrid (GatedDeltaNet) is genuinely slow:
    # weight load + torch.compile + a ~450s profiling/warmup run + CUDA-graph
    # capture + flashinfer GDN JIT ≈ 20 min on first boot (the torch.compile cache
    # speeds *only* compilation on later boots; warmup/capture recur). The old
    # 1200s gate tripped right as the server finished, so we allow 40 min.
    port = cfg["judge"]["port"]
    if _probe(port):
        print(f"[judge] already serving on :{port}")
        return
    _install(cfg)
    subprocess.run(["supervisorctl", "start", SERVICE], check=True)
    print(f"[judge] waiting for {cfg['judge']['model']} on :{port} (model load takes a few minutes)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _probe(port):
            print(f"[judge] ready after {time.time() - t0:.0f}s")
            return
        # Fail fast if supervisor reports the process dead.
        out = subprocess.run(["supervisorctl", "status", SERVICE], capture_output=True, text=True).stdout
        if "FATAL" in out or "EXITED" in out:
            sys.exit(f"[judge] service failed to start — see /var/log/portal/{SERVICE}.log")
        time.sleep(10)
    sys.exit(f"[judge] not ready after {timeout_s}s — see /var/log/portal/{SERVICE}.log")


def down() -> None:
    subprocess.run(["supervisorctl", "stop", SERVICE])


def status(cfg: dict) -> None:
    subprocess.run(["supervisorctl", "status", SERVICE])
    port = cfg["judge"]["port"]
    print(f"[judge] /v1/models on :{port}: {'OK' if _probe(port) else 'not responding'}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("action", choices=["up", "down", "status"])
    args = p.parse_args(argv)
    cfg = load_config()
    {"up": lambda: up(cfg), "down": down, "status": lambda: status(cfg)}[args.action]()


if __name__ == "__main__":
    main()
