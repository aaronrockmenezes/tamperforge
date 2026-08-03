#!/usr/bin/env python
"""Guard the version_B attack mix. Run: python experiments/test_sampler_mix.py

Two things worth failing loudly on:
  1. p_heretic=0.0 must leave the RNG stream untouched, or version_B / version_E / ART stop
     being reproducible from their logged seeds and every archived number loses its provenance.
  2. p_heretic must actually buy write-only mass, since the whole point is that the subset draw
     cannot reach that geometry by reweighting (2/7 at k=1, 1/21 at k=2, impossible at k>=3).
"""
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import version_a_attack as VA  # noqa: E402

NL, N = 28, 200000
# Measured on the sampler as it stood at commit b9cbafb, before --version-b-p-heretic existed.
BASELINE = {"read+write": 67.44, "read-only": 26.67, "write-only": 5.88}


def mix(**kw):
    rng = random.Random(42)
    c = collections.Counter()
    for _ in range(N):
        s = VA.sample_attack_b(rng, NL, **kw)
        w, r = bool(s.write_proj), bool(s.read_proj)
        c["write-only" if (w and not r) else "read-only" if (r and not w) else "read+write"] += 1
        c["tent"] += s.tag.startswith("heretic:tent")
    return {k: 100 * v / N for k, v in c.items()}


def main():
    d = mix()
    for k, want in BASELINE.items():
        assert abs(d[k] - want) < 0.01, f"RNG stream moved: {k} {d[k]:.2f} != {want}"
    assert d.get("tent", 0) == 0.0, "disabled slice still fired"

    d = mix(p_canonical=0.10, p_heretic=0.35)
    assert abs(d["tent"] - 35.0) < 0.5, f"tent share {d['tent']:.1f}, want 35"
    assert d["write-only"] > 38.0, f"write-only {d['write-only']:.1f}, want >38"

    # Tents must stay write-only and land on the stack, or _ablated_overrides silently no-ops.
    rng = random.Random(7)
    for _ in range(20000):
        s = VA.sample_attack_b(rng, NL, p_canonical=0.10, p_heretic=0.5)
        if not s.tag.startswith("heretic:tent"):
            continue
        assert s.read_proj == () and s.write_proj == VA.HERETIC_PROJ, s
        assert s.layers and all(0 <= li < NL for li in s.layers), s
        assert all(isinstance(v, dict) for v in s.alphas.values()), "not per-projection tents"
        assert any(s.alphas[p].get(li, 0.0) > 0 for p in s.write_proj for li in s.layers)

    print("sampler mix OK")


if __name__ == "__main__":
    main()
