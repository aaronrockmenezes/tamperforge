"""Second pass: output guards the first pass missed.

Only OUTPUT-production guards are rewritten. Guards that check an INPUT exists --
`[ -d "$D" ] || { echo "!! missing $D"; continue; }` -- are correct as written and are
left alone.
"""
import re, subprocess, sys

FIND = "find %s -name 'results_*.json' -print -quit 2>/dev/null | grep -q ."
SUBS = [
    # lm_eval output dir, "run unless present"
    (re.compile(r'\[ -d "\$(out|o)" \] \|\| lm_eval'),
     lambda m: (FIND % f'"${m.group(1)}"') + ' || lm_eval'),
    # lm_eval output dir, "skip if present" (with or without an echo)
    (re.compile(r'\[ -d "\$(out|o)" \] && \{ echo [^;]*; continue; \}'),
     lambda m: (FIND % f'"${m.group(1)}"') + ' && continue'),
    (re.compile(r'\[ -d "\$(out|o)" \] && continue'),
     lambda m: (FIND % f'"${m.group(1)}"') + ' && continue'),
    # materialised model dirs from save_p1b_checkpoint
    (re.compile(r'\[ -d "\$2" \] \|\| \$PY experiments/save_p1b_checkpoint\.py'),
     lambda m: '[ -f "$2/model.safetensors" ] || $PY experiments/save_p1b_checkpoint.py'),
]
changed = 0
for path in sys.argv[1:]:
    try:
        src = open(path).read()
    except OSError:
        continue
    out = src
    for pat, rep in SUBS:
        out = pat.sub(rep, out)
    if out != src:
        open(path, "w").write(out)
        rc = subprocess.run(["bash", "-n", path], capture_output=True)
        if rc.returncode != 0:
            open(path, "w").write(src)
            print(f"REVERTED (syntax) {path}")
            continue
        changed += 1
        print(f"fixed {path}")
print(f"{changed} updated")
