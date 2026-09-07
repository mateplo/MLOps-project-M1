#!/usr/bin/env bash
# Release gate: refuses to tag unless code, model and image are all consistent.
#   make release-check [MIN_AUC=0.92] [PORT=8001] [N=200]
set -euo pipefail

MIN_AUC="${MIN_AUC:-0.92}"
PORT="${PORT:-8001}"
N="${N:-200}"
IMAGE="${IMAGE:-adult-income-classifier:latest}"
PY="${PY:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/config.yaml}"
CONTAINER="aic-release-check"

ok()   { printf "  \033[32m✔\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✘\033[0m %s\n" "$*"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; exit 1; }

echo "1. Git"
[ -z "$(git status --porcelain)" ] || fail "working tree not clean (commit or stash first)"
SHA=$(git rev-parse HEAD)
ok "clean tree at ${SHA:0:7}"

echo "2. CI on this commit"
CI=$(curl -sf "https://api.github.com/repos/$(git remote get-url origin | sed -E 's#.*github.com[:/]##; s#\.git$##')/actions/runs?head_sha=$SHA&per_page=1" \
     | "$PY" -c "import sys,json; r=json.load(sys.stdin).get('workflow_runs',[]); print(r[0]['conclusion'] if r else 'none')" 2>/dev/null || echo "unknown")
[ "$CI" = "success" ] || fail "CI conclusion for ${SHA:0:7} is '$CI' (need success)"
ok "CI green"

echo "3. Model = champion from the registry"
"$PY" -W ignore -m src.export --config "$CONFIG" >/dev/null 2>&1 || fail "export failed"
read -r VERSION ALIAS RUN_ID AUC <<<"$("$PY" - <<'PY'
import json; m=json.load(open("artifacts/model_meta.json"))
print(m.get("version"), m.get("alias"), m.get("run_id"), m["metrics"].get("test_roc_auc"))
PY
)"
[ "$ALIAS" = "champion" ] || fail "exported model alias is '$ALIAS', not champion"
ok "AdultIncomeClassifier v$VERSION (champion, run ${RUN_ID:0:8})"

echo "4. Metrics floor"
"$PY" -c "import sys; sys.exit(0 if float('$AUC') >= float('$MIN_AUC') else 1)" \
  || fail "test_roc_auc $AUC < MIN_AUC $MIN_AUC"
ok "test_roc_auc $AUC >= $MIN_AUC"

echo "5. Image serves that exact model"
docker build -q --target serve -t "$IMAGE" . >/dev/null || fail "docker build failed"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" -p "$PORT:8000" -v "$PWD/artifacts:/app/artifacts" "$IMAGE" >/dev/null
for _ in $(seq 1 60); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 0.5; done
HEALTH=$(curl -sf "localhost:$PORT/health") || fail "API not healthy"
SERVED=$("$PY" -c "import json,sys; h=json.loads(sys.argv[1]); print(h.get('model_version'), h.get('run_id'))" "$HEALTH")
[ "$SERVED" = "$VERSION $RUN_ID" ] || fail "API serves '$SERVED', expected 'v$VERSION $RUN_ID'"
ok "container healthy, serving v$VERSION"

echo "6. Traffic smoke ($N real rows, 0 errors expected)"
OUT=$("$PY" -W ignore -m src.simulate_traffic --config "$CONFIG" --url "http://localhost:$PORT" --n "$N" 2>&1 | grep -E "ok," || true)
echo "$OUT" | grep -q " 0 errors" || fail "simulate reported errors: $OUT"
ok "$(echo "$OUT" | sed -E 's/.*INFO +__main__: //')"
docker rm -f "$CONTAINER" >/dev/null

echo
echo "All checks passed. Suggested release:"
echo "  git tag -a vX.Y.Z -m \"AdultIncomeClassifier v$VERSION (test_roc_auc $AUC, run ${RUN_ID:0:8})\" && git push origin vX.Y.Z"
