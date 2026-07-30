"""version_A attack sampler -- adds capability-overlap as a training axis.

NAMING: the vX line stops at v11. New defense work is version_A, version_B, ... The old
ABL-v8/v9/v10/v11 names keep their existing meanings; nothing is renamed retroactively.

WHY THIS EXISTS
---------------
v8's ensemble varies where the attack lands (scope, layer set) and how hard it hits
(per-layer alpha), and optionally whether each layer gets its own direction. It does not
vary the one property the surgical attack exploits: how much of the ablated direction lies
inside the capability subspace.

Measured (results/v11_surgical_overlap/training_direction_overlap.json): across 4 prompt
resamples x 3 read layers, every direction v8 trains against has capability overlap in
[0.769, 0.803] at the rank-16 frame. d_surgical is 0.000. The model has never seen an
ablation below 0.77 and is then attacked at zero -- and loses (44.8% harmful, capability
intact).

So version_A samples the overlap axis explicitly: a fraction of steps ablate
`normalise(d - P_cap d)` instead of `d`, with the capability rank drawn from the range a
real attacker would pick. k=4 is where the escape hatch opens (below it the attacker's own
capability craters), k=16 is the demonstrated break.

WHAT THIS DOES NOT SOLVE
------------------------
The wall/heal oscillation is untouched and independent. And a moving target remains: the
attacker computes d_surgical against FINAL weights, while training sees it recomputed every
refresh_every steps. If the collapse ends up keyed to a stale subspace this axis buys
nothing -- recompute cadence is the knob to watch.

USAGE
    bank = DirectionBank.build(model, tok, device, harmful, benign, cap_prompts,
                               layers=attack_layers, read_layers=(16, 20, 24),
                               cap_ranks=(2, 4, 8, 16), rng=rng_direction)
    spec = sample_attack(rng_attack, n_layers, attack_band=attack_layers)
    d_or_dict = bank.directions_for(spec)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import NamedTuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import capture_residuals, empirical_refusal_directions  # noqa: E402

SCOPES = {
    "all": (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "mlp.gate_proj", "mlp.up_proj"), ("self_attn.o_proj", "mlp.down_proj")),
    "mlp": (("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",)),
    "attn": (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
             ("self_attn.o_proj",)),
}


class AttackSpec(NamedTuple):
    read_proj: tuple[str, ...]
    write_proj: tuple[str, ...]
    layers: list[int]
    alphas: dict[int, float] | None     # None = full strength
    per_layer: bool                     # each layer uses its own direction
    variant: str                        # "plain" | "surgical"
    cap_rank: int                       # 0 when plain
    read_layer: int                     # which layer the shared direction was read from
    tag: str


def sample_attack(
    rng: random.Random,
    n_layers: int,
    *,
    attack_band: list[int] | None = None,
    read_layers: tuple[int, ...] = (16, 20, 24),
    cap_ranks: tuple[int, ...] = (2, 4, 8, 16),
    p_canonical: float = 0.20,
    p_surgical: float = 0.40,
    p_per_layer: float = 0.50,
    p_partial: float = 0.50,
) -> AttackSpec:
    """Sample one training attack.

    p_canonical is drawn FIRST and short-circuits everything else: full-strength,
    shared-direction, all-layer, plain rank-1 -- the standard Arditi attack. The headline
    result is built on it, so it keeps guaranteed gradient mass instead of being diluted to
    near-zero by the product of the other probabilities (v8's sampler docstring makes the
    same point about per-layer).
    """
    band = attack_band or list(range(n_layers))
    read_layer = rng.choice([L for L in read_layers if 0 <= L < n_layers] or [n_layers // 2])

    if rng.random() < p_canonical:
        rp, wp = SCOPES["all"]
        return AttackSpec(rp, wp, list(range(n_layers)), None, False, "plain", 0,
                          read_layer, "canonical:arditi")

    scope = rng.choice(["all", "mlp", "attn"])
    rp, wp = SCOPES[scope]

    kind = rng.choice(["all", "broad", "upper", "rand"])
    if kind == "all":
        layers = list(range(n_layers))
    elif kind == "broad":                       # Heretic shape: nearly all layers
        lo = rng.randint(0, max(0, n_layers // 4))
        layers = list(range(lo, n_layers))
    elif kind == "upper":
        layers = list(range(n_layers // 2, n_layers))
    else:
        lo = rng.choice(band)
        hi = rng.randint(min(lo + n_layers // 2, n_layers - 1), n_layers - 1)
        layers = list(range(lo, hi + 1))
    layers = layers or list(range(n_layers))

    alphas = ({li: rng.uniform(0.2, 1.0) for li in layers}
              if rng.random() < p_partial else None)
    per_layer = rng.random() < p_per_layer

    if rng.random() < p_surgical:
        variant, cap_rank = "surgical", rng.choice(list(cap_ranks))
    else:
        variant, cap_rank = "plain", 0

    tag = (f"{scope}:{kind}{'' if alphas is None else ':partial'}"
           f"{':perlayer' if per_layer else f':L{read_layer}'}"
           f"{'' if variant == 'plain' else f':surg{cap_rank}'}")
    return AttackSpec(rp, wp, layers, alphas, per_layer, variant, cap_rank, read_layer, tag)


def _subspace(H: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-`rank` right singular directions of centred activations.

    Same three lines as v11_surgical_ablation._cap_subspace, kept here so every layer can
    be done from ONE capture pass -- calling that helper per layer re-runs every prompt per
    layer, which is unaffordable at training refresh cadence.
    """
    H = H.float()
    H = H - H.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(H, full_matrices=False)
    return Vh[:rank]


def _surgical(d: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, float]:
    """`normalise(d - P_cap d)` plus the overlap that was removed."""
    d_cap = V.T @ (V @ d)
    overlap = float((d_cap.norm() / d.norm().clamp(min=1e-9)).clamp(0, 1))
    d_s = d - d_cap
    n = d_s.norm()
    if float(n) < 1e-4:
        # d lies entirely in the capability span: surgical ablation is undefined, which is
        # the win condition. Fall back to plain so the step still trains something.
        return d / d.norm().clamp(min=1e-9), 1.0
    return d_s / n, overlap


class DirectionBank(NamedTuple):
    """Every direction an AttackSpec might ask for, from one refresh."""
    plain: dict[int, torch.Tensor]                    # layer -> d
    surgical: dict[tuple[int, int], torch.Tensor]     # (layer, cap_rank) -> d_surgical
    overlap: dict[tuple[int, int], float]             # (layer, cap_rank) -> removed overlap

    @classmethod
    @torch.no_grad()
    def build(cls, model, tok, device, harmful, benign, cap_prompts, *, layers,
              read_layers=(16, 20, 24), cap_ranks=(2, 4, 8, 16)):
        want = sorted(set(list(layers) + list(read_layers)))
        plain = {L: v.float().to(device)
                 for L, v in empirical_refusal_directions(
                     model, tok, harmful, benign, want, device).items()}
        Hc = capture_residuals(model, tok, cap_prompts, want, device)
        surg, ov = {}, {}
        for L in want:
            for r in cap_ranks:
                V = _subspace(Hc[L], r).to(device)
                surg[(L, r)], ov[(L, r)] = _surgical(plain[L], V)
        return cls(plain, surg, ov)

    def directions_for(self, spec: AttackSpec):
        """Return {layer: d} for per-layer attacks, else the single shared d."""
        pick = (lambda L: self.plain[L]) if spec.variant == "plain" \
            else (lambda L: self.surgical[(L, spec.cap_rank)])
        if spec.per_layer:
            return {L: pick(L) for L in spec.layers}
        return pick(spec.read_layer)

    def realized_overlap(self, spec: AttackSpec) -> float:
        """Log this per step -- it is the axis version_A exists to vary."""
        if spec.variant == "plain":
            ls = spec.layers if spec.per_layer else [spec.read_layer]
            return sum(self.overlap.get((L, 16), float("nan")) for L in ls) / len(ls)
        return 0.0


def _selfcheck() -> None:
    torch.manual_seed(0)
    d_model = 64

    # surgical math: output is unit-norm and has no capability component left
    V = torch.linalg.qr(torch.randn(d_model, 8))[0].T
    d = torch.randn(d_model)
    d_s, ov = _surgical(d, V)
    assert abs(float(d_s.norm()) - 1.0) < 1e-5, d_s.norm()
    assert float((V @ d_s).abs().max()) < 1e-5, "capability component survived"
    assert 0.0 <= ov <= 1.0

    # degenerate case: d entirely inside the span -> fall back to plain, overlap 1
    d_in = V.T @ torch.randn(8)
    _, ov_in = _surgical(d_in, V)
    assert ov_in == 1.0, ov_in

    # _subspace matches a direct SVD of centred activations
    H = torch.randn(50, d_model)
    Hc = H - H.mean(0, keepdim=True)
    assert torch.allclose(_subspace(H, 4).abs(), torch.linalg.svd(Hc, full_matrices=False)[2][:4].abs(),
                          atol=1e-4)

    # sampler: every spec well-formed, and the canonical arm keeps its mass
    rng = random.Random(0)
    n_layers, n = 28, 4000
    specs = [sample_attack(rng, n_layers) for _ in range(n)]
    for s in specs:
        assert s.layers and all(0 <= L < n_layers for L in s.layers)
        assert s.alphas is None or all(0.2 <= a <= 1.0 for a in s.alphas.values())
        assert s.variant in {"plain", "surgical"}
        assert (s.cap_rank == 0) == (s.variant == "plain")
    canon = sum(s.tag == "canonical:arditi" for s in specs) / n
    surg = sum(s.variant == "surgical" for s in specs) / n
    assert 0.17 < canon < 0.23, canon
    assert 0.29 < surg < 0.35, surg          # 0.8 * 0.4
    print(f"ok  canonical={canon:.3f}  surgical={surg:.3f}  "
          f"variants={len({s.tag for s in specs})}")


if __name__ == "__main__":
    _selfcheck()
