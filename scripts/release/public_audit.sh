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
check_absent './docs/PUBLIC_RELEASE.md' 'internal public release checklist must not be present in public tree'

if find . -path ./.git -prune -o -name '.env*' -print | grep -q .; then
  echo 'FAIL: .env files and examples must not be present in public tree' >&2
  fail=1
fi

if [ -e skills/creative/hambajuba-dance-director/SKILL.md ] || [ -e src/hambajuba2ba/integrations/hermes/skill/SKILL.md ]; then
  if [ ! -e skills/creative/hambajuba-dance-director/SKILL.md ]; then
    echo 'FAIL: public Hermes skill mirror is missing' >&2
    fail=1
  elif [ ! -e src/hambajuba2ba/integrations/hermes/skill/SKILL.md ]; then
    echo 'FAIL: packaged Hermes source skill is missing' >&2
    fail=1
  elif ! cmp -s skills/creative/hambajuba-dance-director/SKILL.md src/hambajuba2ba/integrations/hermes/skill/SKILL.md; then
    echo 'FAIL: public Hermes skill mirror differs from packaged source skill' >&2
    fail=1
  fi
fi

if git grep --untracked -n -I -E '/Users/omarhammami|RunPod|root@|id_ed25519|GITHUB_PAT|OPENROUTER_API_KEY=.*' -- . ':!scripts/release/public_audit.sh'; then
  echo 'FAIL: found private deployment/local path marker' >&2
  fail=1
fi

if git grep --untracked -n -I -E 'personal GPU|personal website|psite' -- . ':!scripts/release/public_audit.sh'; then
  echo 'FAIL: found internal personal release wording' >&2
  fail=1
fi

if git grep --untracked -n -I -E 'sk-or-v1-|ghp_|hf_[A-Za-z0-9]{20,}' -- . ':!scripts/release/public_audit.sh'; then
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
