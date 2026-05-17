#!/usr/bin/env bash
# test-regenerate-memory-index.sh — smoke tests for regenerate-memory-index.py
#
# Covers the issue #787 regression: regen running with 0 frontmatter siblings
# must NOT clobber an existing non-empty MEMORY.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
SCRIPT="skills/cross-node-sync/scripts/regenerate-memory-index.py"

PASS=0
FAIL=0
pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "━━━ regenerate-memory-index.py smoke tests ━━━"

# T1 — python parse
if python3 -c "import ast; ast.parse(open('$SCRIPT').read())" 2>/dev/null; then
    pass "T1: python parses clean"
else
    fail "T1: python syntax error"
fi

# T2 — issue #787: refuse to write when 0 entries + existing non-empty MEMORY.md
MEMDIR="$TMP/case_refuse"; mkdir -p "$MEMDIR"
printf '# index\n\n# userEmail\nfoo@bar\n' > "$MEMDIR/MEMORY.md"
EXISTING_HASH="$(shasum "$MEMDIR/MEMORY.md" | cut -d' ' -f1)"
OUT="$(python3 "$SCRIPT" --memory-dir "$MEMDIR" 2>&1 || true)"
NEW_HASH="$(shasum "$MEMDIR/MEMORY.md" | cut -d' ' -f1)"
if [[ "$EXISTING_HASH" == "$NEW_HASH" ]] && echo "$OUT" | grep -q "refuse"; then
    pass "T2: refuses to clobber non-empty MEMORY.md when 0 entries found (#787)"
else
    fail "T2: MEMORY.md was modified or no refuse message"
    echo "    output: $OUT"
fi

# T3 — happy path: regen writes index when frontmatter files exist
MEMDIR="$TMP/case_write"; mkdir -p "$MEMDIR"
cat > "$MEMDIR/foo.md" <<'EOF'
---
name: foo
description: a foo memory
metadata:
  type: user
---
body
EOF
python3 "$SCRIPT" --memory-dir "$MEMDIR" >/dev/null 2>&1
if grep -q "\[foo\](foo.md) — a foo memory" "$MEMDIR/MEMORY.md"; then
    pass "T3: writes index entry from frontmatter"
else
    fail "T3: expected entry not in MEMORY.md"
    cat "$MEMDIR/MEMORY.md"
fi

# T4 — fresh dir (no MEMORY.md, no entries) writes empty body without error
MEMDIR="$TMP/case_fresh"; mkdir -p "$MEMDIR"
python3 "$SCRIPT" --memory-dir "$MEMDIR" >/dev/null 2>&1
if [[ -f "$MEMDIR/MEMORY.md" ]]; then
    pass "T4: fresh dir gets empty MEMORY.md created"
else
    fail "T4: MEMORY.md not created on fresh dir"
fi

# T5 — dry-run does not write
MEMDIR="$TMP/case_dryrun"; mkdir -p "$MEMDIR"
printf 'preexisting\n' > "$MEMDIR/MEMORY.md"
python3 "$SCRIPT" --memory-dir "$MEMDIR" --dry-run >/dev/null 2>&1
if [[ "$(cat "$MEMDIR/MEMORY.md")" == "preexisting" ]]; then
    pass "T5: --dry-run leaves file untouched"
else
    fail "T5: --dry-run modified the file"
fi

echo "━━━ $PASS passed / $FAIL failed ━━━"
[[ $FAIL -eq 0 ]]
