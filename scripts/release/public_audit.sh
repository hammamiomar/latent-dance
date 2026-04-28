#!/bin/bash
# Public tree sanity checks. Run from repo root.
set -euo pipefail

fail=0

check_absent() {
  local pattern="$1"
  local message="$2"
  if find . -path ./.git -prune -o -path "$pattern" -print | grep -q .; then
    echo "FAIL: $message" >&2
    fail=1
  fi
}

check_absent './notes*' 'notes/ must not be present in public tree'
check_absent './private*' 'private/ must not be present in public tree'
check_absent './data/sdxl/sae_weights*' 'SAE weights must not be present in public tree'

if git grep -n -I -E '/Users/omarhammami|RunPod|root@|id_ed25519|GITHUB_PAT|OPENROUTER_API_KEY=.*' -- . ':!scripts/release/public_audit.sh'; then
  echo 'FAIL: found private deployment/local path marker' >&2
  fail=1
fi

if git grep -n -I -E 'sk-or-v1-|ghp_|hf_[A-Za-z0-9]{20,}' -- . ':!scripts/release/public_audit.sh'; then
  echo 'FAIL: possible secret token found' >&2
  fail=1
fi

if git grep -n -I 'HAMBA_ARTIFACT_REPO=hammamiomar/sdxl-turbo-sae-labels' -- Dockerfile .github scripts ':!scripts/release/public_audit.sh' 2>/dev/null; then
  echo 'FAIL: Docker runtime points at label dataset instead of upstream SAE weights' >&2
  fail=1
fi

if git ls-files | grep -E '(^|/)\.DS_Store$'; then
  echo 'FAIL: .DS_Store is tracked' >&2
  fail=1
fi

exit "$fail"
