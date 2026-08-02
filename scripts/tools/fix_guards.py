"""Replace directory-existence guards with artifact-existence guards across eval scripts.

`[ -d results/X ]` is true for a directory a killed job left behind with nothing in it, so
the rerun skips regeneration and the judge then runs on a missing generations.jsonl. This
silently destroyed four eval arms on 2026-08-01/02 (lbase_clean, rep_vb_s1, rep_vc_s1, and
the earlier pair). Guard on the ARTIFACT the step produces, never on the directory.

  p0_baseline_eval  -> results/<id>/generations.jsonl
  judge_generations -> results/<id>_judged/summary.json
  lm_eval           -> results/<out>/**/results_*.json   (nested, needs find)
  model export      -> <dir>/model.safetensors
"""
import re, subprocess, sys

SCRIPTS = sys.argv[1:]
SUBS = [
    # judge: dir -> summary.json
    (re.compile(r'\[ -d "(results/[^"]*_judged)" \]'), r'[ -f "\1/summary.json" ]'),
    # p0 generations: dir -> generations.jsonl  (only when NOT already _judged)
    (re.compile(r'\[ -d "(results/(?:\$\{TAG\}|[^"]*?))" \](?! \|\| \{)'),
     r'[ -f "\1/generations.jsonl" ]'),
    # exported model dirs: dir -> model.safetensors
    (re.compile(r'\[ -d "(\$D|\$\{?D\}?)" \] \|\| python'), r'[ -f "\1/model.safetensors" ] || python'),
]
LM = re.compile(r'\[ -d "\$o" \] && continue')
LM_NEW = ('find "$o" -name \'results_*.json\' -print -quit 2>/dev/null | grep -q . && continue')

changed = 0
for path in SCRIPTS:
    try:
        src = open(path).read()
    except OSError:
        continue
    out = src
    for pat, rep in SUBS:
        out = pat.sub(rep, out)
    out = LM.sub(LM_NEW, out)
    if out != src:
        open(path, "w").write(out)
        rc = subprocess.run(["bash", "-n", path], capture_output=True)
        if rc.returncode != 0:
            open(path, "w").write(src)   # revert on syntax error
            print(f"REVERTED (syntax) {path}: {rc.stderr.decode()[:80]}")
            continue
        changed += 1
        print(f"fixed {path}")
print(f"\n{changed} script(s) updated")
