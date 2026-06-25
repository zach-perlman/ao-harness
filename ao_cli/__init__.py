"""ao_cli — thin orchestration layer around the vendored activation_oracles repo.

Gears-level overview:
  config.yaml is the single source of truth. Each subcommand (see __main__.py)
  reads it, derives concrete artifacts (training-config JSON, CLI invocations,
  dataset paths) and shells out to the right virtualenv. Nothing in here does
  ML work itself — it only wires together the vendored code.

Layout conventions (all relative to PROJECT_ROOT = /workspace/ao):
  activation_oracles/        vendored repo (training + data pipelines)
  activation_oracles/third_party/cot-oracle/AObench   eval suite
  envs/{train,vllm,legacy}/  uv virtualenvs
  artifacts/<model-slug>/    corpus, convqa, train configs, checkpoints, evals
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO = PROJECT_ROOT / "activation_oracles"
AOBENCH_ROOT = REPO / "third_party" / "cot-oracle"
ARTIFACTS = PROJECT_ROOT / "artifacts"
ENVS = PROJECT_ROOT / "envs"

PY_TRAIN = ENVS / "train" / "bin" / "python"
PY_VLLM = ENVS / "vllm" / "bin" / "python"
PY_LEGACY = ENVS / "legacy" / "bin" / "python"


def load_config() -> dict:
    """Load config.yaml, then apply optional per-run layer overrides from the env.

    AO_CENTER_PERCENT / AO_N_LAYERS let a sweep driver vary the injection depth
    cell-by-cell without rewriting config.yaml (the same role AO_EXP plays for the
    run name). Unset = use the file's values. This is the only train-time knob a
    layer sweep needs, because `layers` is read here by every subcommand.
    """
    with open(PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    for env_key, cfg_key in (("AO_CENTER_PERCENT", "center_percent"), ("AO_N_LAYERS", "n_layers")):
        if os.environ.get(env_key, "").strip():
            cfg["layers"][cfg_key] = int(os.environ[env_key])
    return cfg


def model_slug(model_name: str) -> str:
    """'Qwen/Qwen3.5-4B' -> 'Qwen3.5-4B' (matches the repo's model_dir_name)."""
    return model_name.split("/")[-1]


def artifacts_dir(model_name: str) -> Path:
    d = ARTIFACTS / model_slug(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def num_hidden_layers(model_name: str) -> int:
    """Read num_hidden_layers from the model's HF config.json (handles the
    nested text_config used by multimodal architectures like Qwen3.5)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_name, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    if "num_hidden_layers" in cfg:
        return cfg["num_hidden_layers"]
    return cfg["text_config"]["num_hidden_layers"]


def resolve_layers(model_name: str, center_percent: int, n_layers: int) -> tuple[list[int], list[int]]:
    """Map (center %, count) -> (absolute contiguous layer indices, percent labels).

    Mechanism: center = round(L * pct/100); take n contiguous layers around it,
    clamped to [1, L-2] so we never read the embedding or final layer.

    The percent labels are the STORED form (config.layer_combinations); the
    decoders that recover the absolute layer (nl_probes + AObench
    `layer_percent_to_layer`) MUST use round(), the exact inverse of the round()
    encode below — they round-trip for any model with <100 layers. (The vendored
    AObench copy historically used int()/floor, which desynced eval from training
    and collapsed adjacent labels onto a duplicate layer; that copy is now round.)
    """
    L = num_hidden_layers(model_name)
    center = round(L * center_percent / 100)
    half = n_layers // 2
    start = max(1, min(center - half, L - 2 - (n_layers - 1)))
    layers = list(range(start, start + n_layers))
    percents = [round(l / L * 100) for l in layers]
    return layers, percents


def judge_base_url(cfg: dict) -> str:
    return f"http://127.0.0.1:{cfg['judge']['port']}/v1"


def experiment_name() -> str:
    """The active experiment name, from the EXP make var (threaded via AO_EXP).

    Empty string = the default run. When set, it namespaces BOTH the training
    checkpoint (checkpoints/<EXP>/final) and the eval results dir
    (aobench_results/<EXP>), so each contribution is an isolated run that the
    dashboard compares against `baseline_replication` without ever clobbering
    `main` or the baseline. See contributions.* in config.yaml.
    """
    return os.environ.get("AO_EXP", "").strip()


def run_dir_name(model_name: str, smoke: bool = False) -> str:
    """Checkpoint/run-dir base name: <EXP> if set, else the default ao_<slug>_v2."""
    if smoke:
        return f"ao_{model_slug(model_name)}_v2_smoke"
    return experiment_name() or f"ao_{model_slug(model_name)}_v2"


def _terminate_group(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """Blocking, escalating teardown of a child's ENTIRE process group.

    SIGTERM the group, wait up to `grace` for it to release the GPU, then SIGKILL
    survivors. Safe ONLY outside a signal handler — it calls proc.wait(), which
    is not re-entrant against Popen's internal waitpid lock (see _forward).
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def run(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> None:
    """Run a subprocess with inherited stdio; exit on failure (fail-fast).

    The vendored scripts import their own top-level packages (`nl_probes.*`,
    `data_pipelines.*`) but are launched as path scripts, so Python puts only
    the script's directory on sys.path — not the package root. We add the
    working directory (the package root for each invocation) to PYTHONPATH so
    those absolute imports resolve regardless of how the script is invoked.

    Signal handling: the child runs in its OWN session, and we install SIGINT/
    SIGTERM handlers that forward a full-group teardown to it. So a Ctrl+C in the
    terminal (or a SIGTERM to this orchestrator) reliably stops the *whole*
    training tree — torchrun and all its CUDA workers — instead of killing only
    this wrapper and leaving the trainer orphaned on the GPU.
    """
    workdir = cwd or REPO
    base_pp = (env or {}).get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    pythonpath = f"{workdir}:{base_pp}".rstrip(":")
    full_env = {**os.environ, **(env or {}), "PYTHONPATH": pythonpath}
    printable = " ".join(str(c) for c in cmd)
    print(f"\n[ao] $ {printable}\n", flush=True)

    proc = subprocess.Popen(
        [str(c) for c in cmd], env=full_env, cwd=str(workdir), start_new_session=True
    )
    state: dict[str, int | None] = {"sig": None}

    def _forward(signum, _frame):
        # Runs in the main thread WHILE it is blocked in proc.wait() below, so it
        # must not call proc.wait()/poll() (re-entrant -> deadlock on Popen's
        # waitpid lock). Instead: SIGTERM the whole group now, arm a daemon timer
        # to escalate to SIGKILL, and let the blocked proc.wait() return once the
        # group dies. The orchestrator then exits with the signal's code.
        state["sig"] = signum
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return

        def _hard_kill():
            time.sleep(10)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        threading.Thread(target=_hard_kill, daemon=True).start()

    prev_int = signal.signal(signal.SIGINT, _forward)
    prev_term = signal.signal(signal.SIGTERM, _forward)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        # Fallback if SIGINT ever arrives as a Python KeyboardInterrupt rather
        # than via _forward; here we are not inside the handler, so blocking is OK.
        _terminate_group(proc)
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
    if state["sig"] is not None:
        raise SystemExit(128 + state["sig"])
    if returncode != 0:
        sys.exit(returncode)
