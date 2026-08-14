"""Every entrypoint's argparse must actually build.

WHY THIS EXISTS. `--version-b-jitter-deg` shipped defined TWICE, which makes argparse raise
`ArgumentError: conflicting option string` before the trainer does anything at all. Nothing local
caught it: pytest never parses args, the chain's DRY_RUN never invokes the trainer, and the
selftests exercise library functions rather than main(). It surfaced only on a rented GPU box,
during the setup smoke test, after the repo had been rsynced and the caches warmed.

`--help` is the cheapest thing that builds the whole parser and imports the whole module, so it
catches duplicate flags, import errors and syntax errors in one subprocess per entrypoint.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    "experiments/train_tamper_resistant_v8.py",
    "experiments/save_p1b_checkpoint.py",
    "experiments/judge_generations.py",
    "scripts/probes/posthoc_lrr.py",
    "scripts/probes/gamma_surgical_amplification.py",
    "scripts/probes/rr_gradient_scale.py",
    "scripts/probes/arch_scout.py",
]


@pytest.mark.parametrize("rel", ENTRYPOINTS)
def test_help_builds(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, str(path), "--help"],
                       capture_output=True, text=True, timeout=180, cwd=ROOT, env=env)
    assert r.returncode == 0, f"{rel} --help exited {r.returncode}:\n{r.stderr[-1500:]}"
