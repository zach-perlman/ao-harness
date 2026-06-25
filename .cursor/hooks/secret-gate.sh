#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Secret-commit gate (Cursor beforeShellExecution hook)
#
# PURPOSE: deterministically block `git commit` / `git push` from the agent
#   when the staged (or outgoing) ADDED lines contain something that looks like
#   an API key / token / private key. Runs outside the model => costs 0 tokens.
#
# HOW IT WORKS:
#   1. Read the hook's JSON on stdin; pull out the shell command.
#   2. If it isn't a git commit/push, allow immediately.
#   3. Build the set of *added* lines: `git diff --cached` for a commit, plus
#      the outgoing range (upstream..HEAD) for a push when an upstream exists.
#   4. Grep those additions for known secret prefixes (specific patterns to
#      avoid false positives like the literal word "wandb_project").
#   5. If any hit -> return {"permission":"deny"} with a masked preview, which
#      blocks the command. Otherwise allow.
#
# failClosed:true in hooks.json means a crash/timeout also blocks (fail-safe).
# We avoid `set -e` so the script always reaches an explicit allow/deny.
# ---------------------------------------------------------------------------

input="$(cat)"
cmd="$(printf '%s' "$input" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("command",""))' 2>/dev/null || true)"

allow() { printf '{"permission":"allow"}'; exit 0; }
deny() { # $1 = masked hit list
  python3 - "$1" <<'PY'
import json, sys
hits = sys.argv[1]
print(json.dumps({
    "permission": "deny",
    "user_message": "Secret-gate blocked this git command: the diff contains likely API keys/secrets:\n" + hits + "\nScrub them (use env vars) and retry.",
    "agent_message": "A pre-commit hook detected likely secrets in the staged/outgoing diff and BLOCKED the commit/push. Do NOT retry as-is: replace the hardcoded keys with os.environ lookups, then commit again.",
}))
PY
  exit 0
}

case "$cmd" in
  *"git commit"*|*"git push"*) : ;;
  *) allow ;;
esac

# Gather additions to inspect.
diff="$(git diff --cached -U0 2>/dev/null || true)"
if printf '%s' "$cmd" | grep -q 'git push'; then
  up="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  [ -n "$up" ] && diff="$diff
$(git log "$up"..HEAD -p -U0 2>/dev/null || true)"
fi

added="$(printf '%s' "$diff" | grep -E '^\+' | grep -Ev '^\+\+\+' || true)"

# Specific prefixes only (high precision, low false-positive).
patterns='sk-or-v1-[A-Za-z0-9]{20}|sk-ant-[A-Za-z0-9_-]{20}|sk-[A-Za-z0-9]{32}|hf_[A-Za-z0-9]{30}|ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10}|-----BEGIN [A-Z ]*PRIVATE KEY-----|WANDB_API_KEY[^A-Za-z0-9]{0,20}[0-9a-f]{40}'

hits="$(printf '%s' "$added" | grep -oE "$patterns" | sort -u | head -20 || true)"
if [ -n "$hits" ]; then
  masked="$(printf '%s' "$hits" | sed -E 's/(.{10}).*(.{4})$/\1…\2/')"
  deny "$masked"
fi
allow
