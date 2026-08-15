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

import math
import random
import sys
from pathlib import Path
from typing import NamedTuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import (RANK_ESTIMATORS, capture_residuals,
                         refusal_subspaces_from_activations)  # noqa: E402

VERSION_G_ATTACK_RANKS = (1, 2, 4, 8, 16)

SCOPES = {
    "all": (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "mlp.gate_proj", "mlp.up_proj"), ("self_attn.o_proj", "mlp.down_proj")),
    "mlp": (("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",)),
    "attn": (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
             ("self_attn.o_proj",)),
}


class Tent(NamedTuple):
    """Heretic's per-projection weight profile: a linear tent with a hard cutoff.

    Peaks at `max_weight` on layer `max_pos` (fractional), falls linearly to
    `min_weight` at distance `min_dist`, and is exactly 0 beyond that. Heretic gives
    `attn.o_proj` and `mlp.down_proj` INDEPENDENT tents -- own centre, own width, own
    floor -- which is the axis version_B could not draw (it shared one band and one
    alpha dict across every projection). See docs/handoff_2026_08_01_version_a_b.md 2b.
    """
    max_weight: float
    max_pos: float
    min_weight: float
    min_dist: float


def tent_weight(t: Tent, layer: float) -> float:
    d = abs(float(layer) - t.max_pos)
    if d > t.min_dist or t.min_dist <= 0:
        return 0.0
    return t.max_weight - (t.max_weight - t.min_weight) * (d / t.min_dist)


def tent_alphas(profiles: dict[str, Tent], n_layers: int) -> tuple[list[int], dict[str, dict[int, float]]]:
    """Materialise {proj: {layer: weight}} plus the union of layers with any nonzero weight."""
    out: dict[str, dict[int, float]] = {}
    touched: set[int] = set()
    for name, t in profiles.items():
        w = {li: tent_weight(t, li) for li in range(n_layers)}
        w = {li: v for li, v in w.items() if v > 0.0}
        out[name] = w
        touched |= set(w)
    return sorted(touched), out


# Heretic ablates write-projections only.
HERETIC_PROJ = ("self_attn.o_proj", "mlp.down_proj")
_HERETIC_KEY = {"self_attn.o_proj": "attn.o_proj", "mlp.down_proj": "mlp.down_proj"}


def heretic_spec(params: dict, n_layers: int, *, tag: str = "heretic:replay") -> AttackSpec:
    """Turn a Heretic trial's logged parameters into an AttackSpec we can replay.

    `params` is one entry of results/version_b_final_2026_08_01/summary.json's
    `heretic_trials`. This is the control gate for version_C: if replaying t99 does not
    reproduce its known 0.3212 harmful on version_B s500, the tent implementation is wrong
    and nothing downstream can be trusted.
    """
    profiles = {}
    for proj, key in _HERETIC_KEY.items():
        profiles[proj] = Tent(
            max_weight=float(params[f"{key}.max_weight"]),
            max_pos=float(params[f"{key}.max_weight_position"]),
            min_weight=float(params[f"{key}.min_weight"]),
            min_dist=float(params[f"{key}.min_weight_distance"]),
        )
    layers, alphas = tent_alphas(profiles, n_layers)
    di = params["direction_index"]
    per_layer = (str(di).strip() == "per layer")
    read_layer = 0.0 if per_layer else float(di)
    return AttackSpec((), HERETIC_PROJ, layers, alphas, per_layer,
                      "plain", 0, read_layer, tag)


class AttackSpec(NamedTuple):
    read_proj: tuple[str, ...]
    write_proj: tuple[str, ...]
    layers: list[int]
    # {layer: strength}, or {proj_name: {layer: strength}} for independent
    # per-projection tents (version_C / Heretic replay). None = full strength.
    alphas: dict | None
    per_layer: bool                     # each layer uses its own direction
    variant: str                        # "plain" | "surgical"
    cap_rank: int                       # 0 when plain
    read_layer: float                   # layer the shared direction is read from; version_B
                                        # emits FRACTIONAL values (lerped between neighbours)
    tag: str
    # version_C axes. Defaulted so every existing constructor keeps its meaning:
    # v8/version_A/version_B all trained against the plain application and our own
    # direction recipe, which is exactly what these defaults encode.
    application: str = "plain"          # "plain" | "full" (heretic's row_normalization)
    dir_recipe: str = "ours"            # "ours" | "heretic" (projected abliteration etc.)
    # Direction augmentation. 0.0 = every prior run, bit-identical (see _jitter: it returns
    # before touching rng). Max rotation in DEGREES applied to whatever direction this spec
    # resolves to. Reference points: version_G-Qwen generalises 24.3 deg and fires,
    # version_G-gemma needs 41.0 and does not.
    jitter_deg: float = 0.0
    attack_rank: int = 1                # refusal-subspace rank; cap_rank is separate


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
        # lo is clamped to the lower half so the window always spans >= half the layers.
        # v8's sampler does the same, and for a reason: tier-1 found the collapse is
        # localized and sub-layer ablations under-trigger it, so a 2-layer window is a
        # wasted step that teaches the model nothing about the attack it must survive.
        lo = min(rng.choice(band), n_layers // 2)
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


ALL_PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_WRITE = {"self_attn.o_proj", "mlp.down_proj"}
# small subsets first: a uniform draw over all 127 non-empty subsets concentrates mass on
# 3-4 element ones, and the low-collateral configurations an optimising attacker actually
# picks are the small ones. Heretic's is size 2 ({o_proj, down_proj}).
_SUBSET_SIZE_W = (0.22, 0.22, 0.16, 0.13, 0.11, 0.09, 0.07)


def sample_attack_b(rng: random.Random, n_layers: int, *, jitter_deg: float = 0.0, **kw):
    """version_B sampler + direction augmentation.

    Wraps the sampler instead of threading `jitter_deg` through its three return points --
    one place to set it means no return path can silently miss it. `_replace` is applied
    AFTER sampling, so the RNG stream is untouched and jitter_deg=0.0 reproduces exactly.
    """
    return _sample_attack_b(rng, n_layers, **kw)._replace(jitter_deg=jitter_deg)


def _sample_attack_b(
    rng: random.Random,
    n_layers: int,
    *,
    cap_ranks=(2, 4, 8, 16),
    p_canonical: float = 0.20,
    p_surgical: float = 0.40,
    p_per_layer: float = 0.50,
    p_partial: float = 0.50,
    p_heretic: float = 0.0,
    alpha_max: float = 1.5,
    dir_lo_frac: float = 0.25,
    dir_hi_frac: float = 0.95,
    min_band: int = 2,
    canonical_ranks=(1,),
) -> AttackSpec:
    """version_B. Same skeleton as sample_attack, three axes widened to cover real Heretic.

    Measured gaps that motivate each change (see docs / commit 08b862d):

    - MATRIX SUBSET. version_A had three fixed scopes, every one containing read
      projections, so write-only was never once shown in 500 steps -- and write-only
      (o_proj, down_proj) is exactly and only what Heretic touches. Now any non-empty
      subset of the seven projections.
    - FRACTIONAL, WIDE DIRECTION LAYER. version_A read from {16,20,24}; Heretic searches
      [0.4, 0.9] x last_layer and lerps between neighbours, with its best trials at
      12.6-14.2 -- below version_A's lowest. Range here is deliberately WIDER than
      Heretic's so this does not overfit to one tool's search box.
    - ALPHA TO 1.5. version_A capped at full projection; Heretic samples max_weight in
      [0.8, 1.5], over-projecting past orthogonal. Nothing in 200 trials exceeded 1.5.

    Narrow layer bands are allowed here (min_band=2), unlike version_A which clamped to
    half the stack. Heretic's min_weight_distance goes down to ~1.0, so tight bands are
    in its search space; version_A's clamp was protecting against under-triggering, which
    is the behaviour we now want to train against rather than avoid.

    `p_heretic` (default 0.0, so every pre-2026-08-03 run is bit-identical) carves out an
    explicit slice of Heretic-SHAPED attacks: write-only, independent per-projection tents,
    exactly the geometry `heretic_spec` replays. It exists because the subset draw reaches
    that geometry almost never -- measured over 200k draws at n_layers=28, write-only is
    5.9% of samples and write-only-with-near-full-stack is 0.04%, i.e. ~0.2 steps in a
    500-step run, while Heretic is write-only on EVERY trial. The subset draw cannot fix
    this by reweighting because write-only needs `chosen` to fall inside a 2-element set out
    of 7 (2/7 at k=1, 1/21 at k=2, impossible at k>=3).

    Direction recipe stays ours ("plain"/"ours"), so this changes the attack GEOMETRY only
    and stays comparable with version_B. Note version_C already ran write-only at ~60% and
    regressed; this knob is for controlled placement, not for re-running that.
    """
    dir_layer = rng.uniform(dir_lo_frac * (n_layers - 1), dir_hi_frac * (n_layers - 1))

    if rng.random() < p_canonical:
        rp = tuple(p for p in ALL_PROJ if p not in _WRITE)
        wp = tuple(p for p in ALL_PROJ if p in _WRITE)
        ranks = tuple(canonical_ranks)
        rank = 1 if ranks == (1,) else rng.choice(ranks)
        read_layer = float(round(dir_layer)) if rank > 1 else dir_layer
        tag = "canonical:arditi" if ranks == (1,) else f"canonical:rank_k:k{rank}"
        return AttackSpec(rp, wp, list(range(n_layers)), None, False, "plain", 0,
                          read_layer, tag, attack_rank=rank)

    # Heretic-shaped slice. Conditional on canonical having missed, so the absolute share is
    # p_heretic. Falls through to the subset draw if every tent lands off the stack.
    if p_heretic > 0.0 and rng.random() < p_heretic / max(1e-9, 1.0 - p_canonical):
        profiles = {
            proj: Tent(max_weight=rng.uniform(0.5, alpha_max),
                       max_pos=rng.uniform(dir_lo_frac * (n_layers - 1),
                                           dir_hi_frac * (n_layers - 1)),
                       min_weight=rng.uniform(0.0, 0.9),
                       min_dist=rng.uniform(1.0, 0.6 * n_layers))
            for proj in HERETIC_PROJ
        }
        h_layers, h_alphas = tent_alphas(profiles, n_layers)
        if h_layers:
            per_layer = rng.random() < p_per_layer
            return AttackSpec((), HERETIC_PROJ, h_layers, h_alphas, per_layer, "plain", 0,
                              0.0 if per_layer else dir_layer,
                              f"heretic:tent:L{dir_layer:.1f}:{h_layers[0]}-{h_layers[-1]}"
                              + (":perlayer" if per_layer else ""))

    k = rng.choices(range(1, len(ALL_PROJ) + 1), weights=_SUBSET_SIZE_W)[0]
    chosen = rng.sample(ALL_PROJ, k)
    rp = tuple(p for p in chosen if p not in _WRITE)
    wp = tuple(p for p in chosen if p in _WRITE)

    lo = rng.randint(0, max(0, n_layers - min_band))
    hi = rng.randint(min(lo + min_band - 1, n_layers - 1), n_layers - 1)
    layers = list(range(lo, hi + 1))

    alphas = ({li: rng.uniform(0.2, alpha_max) for li in layers}
              if rng.random() < p_partial else None)
    per_layer = rng.random() < p_per_layer
    if rng.random() < p_surgical:
        variant, cap_rank = "surgical", rng.choice(list(cap_ranks))
    else:
        variant, cap_rank = "plain", 0

    scope_tag = ("w" if wp else "") + ("r" if rp else "") + str(k)
    tag = (f"{scope_tag}:L{dir_layer:.1f}:{lo}-{hi}"
           f"{'' if alphas is None else ':partial'}"
           f"{':perlayer' if per_layer else ''}"
           f"{'' if variant == 'plain' else f':surg{cap_rank}'}")
    return AttackSpec(rp, wp, layers, alphas, per_layer, variant, cap_rank, dir_layer, tag)


def _subspace(H: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-`rank` right singular directions of centred activations.

    Same three lines as v11_surgical_ablation._cap_subspace, kept here so every layer can
    be done from ONE capture pass -- calling that helper per layer re-runs every prompt per
    layer, which is unaffordable at training refresh cadence.
    """
    return _subspaces(H, (rank,))[rank]


def _subspaces(H: torch.Tensor, ranks) -> dict[int, torch.Tensor]:
    """All requested ranks from ONE decomposition -- they are nested prefixes of the same
    basis, so factorising per rank would repeat identical work at every refresh."""
    H = H.float()
    H = H - H.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(H, full_matrices=False)
    return {r: Vh[:r] for r in ranks}


def _refusal_subspaces(
    Hh: torch.Tensor,
    Hb: torch.Tensor,
    ranks,
    estimator: str,
) -> dict[int, torch.Tensor]:
    """Nested rank-k bases; estimator choice is explicit because it changes k=1."""
    return refusal_subspaces_from_activations(Hh, Hb, ranks, estimator)


def _jitter(d: torch.Tensor, deg: float, rng: random.Random | None) -> torch.Tensor:
    """Rotate `d` by a random angle in [0, deg] toward a random orthogonal direction.

    WHY. The defense generalises from the direction it trained on to a neighbouring one only as
    far as it was ever asked to. Measured at cap-rank 16
    (results/gamma_surgical_amplification.json): Qwen version_G must cover 24.3 degrees and
    fires; gemma version_G must cover 41.0 and does not. Training against the discrete endpoints
    {plain, surgical@2/4/8/16} teaches those points, not the arc between and around them.

    Isotropic, not an interpolation toward d_surgical: a real attacker's estimator (mean-diff on
    a different prompt set, SVD-top1, a probe weight vector) lands somewhere in the ball around
    d, not on the segment to any particular d_surgical. Rotating toward a random orthogonal
    covers that ball, and the surgical variants remain sampled separately, so this ADDS coverage
    rather than replacing the axis version_A exists to vary.

    deg <= 0 returns `d` unchanged WITHOUT touching rng, so every existing run reproduces
    bit-identically -- the RNG stream is shared with the attack sampler.
    """
    if deg <= 0 or rng is None:
        return d
    g = torch.empty_like(d).normal_(generator=torch.Generator(device=d.device).manual_seed(
        rng.getrandbits(63)))
    g = g - (g @ d) * d                       # component orthogonal to d
    n = g.norm()
    if float(n) < 1e-8:                       # d was (near) parallel to the draw; skip
        return d
    theta = math.radians(rng.uniform(0.0, deg))
    out = math.cos(theta) * d + math.sin(theta) * (g / n)
    return out / out.norm().clamp(min=1e-9)


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
    ranked: dict[tuple[int, int], torch.Tensor]       # (layer, attack_rank) -> SVD basis
    surgical: dict[tuple[int, int], torch.Tensor]     # (layer, cap_rank) -> d_surgical
    overlap: dict[tuple[int, int], float]             # (layer, cap_rank) -> removed overlap

    @classmethod
    @torch.no_grad()
    def build(cls, model, tok, device, harmful, benign, cap_prompts, *, layers,
              read_layers=(16, 20, 24), cap_ranks=(2, 4, 8, 16), attack_ranks=(),
              rank_estimator="svd"):
        """`layers` must cover EVERY layer the sampler can emit, not just the attack band.

        sample_attack's "all" and "broad" shapes reach outside the band, and a per-layer
        attack then asks the bank for a layer it never built -- a KeyError mid-training.
        Pass range(n_layers).
        """
        if not len(cap_prompts):
            # Empty here surfaces as "torch.cat(): expected a non-empty list of Tensors"
            # from inside capture_residuals, which says nothing about the real cause: the
            # caller did not populate cap_prompts for this attack profile.
            raise ValueError(
                "DirectionBank.build got an empty cap_prompts. The surgical variant needs "
                "capability prompts to estimate the subspace -- check the trainer populated "
                "va_cap_prompts for this --attack-profile.")
        want = sorted(set(list(layers) + list(read_layers)))
        Hh = capture_residuals(model, tok, harmful, want, device)
        Hb = capture_residuals(model, tok, benign, want, device)
        plain, ranked = {}, {}
        attack_ranks = tuple(sorted(set(attack_ranks)))
        for L in want:
            d = Hh[L].mean(0) - Hb[L].mean(0)
            plain[L] = (d / d.norm().clamp(min=1e-9)).to(device)
            if attack_ranks:
                for r, R in _refusal_subspaces(
                    Hh[L], Hb[L], attack_ranks, rank_estimator
                ).items():
                    ranked[(L, r)] = R.to(device)
        Hc = capture_residuals(model, tok, cap_prompts, want, device)
        surg, ov = {}, {}
        for L in want:
            Vs = _subspaces(Hc[L], cap_ranks)
            for r in cap_ranks:
                surg[(L, r)], ov[(L, r)] = _surgical(plain[L], Vs[r].to(device))
        return cls(plain, ranked, surg, ov)

    def direction_at(self, layer: float, cap_rank: int = 0) -> torch.Tensor:
        """Direction at a FRACTIONAL layer, lerped between neighbours then renormalised.

        Mirrors heretic/model.py::abliterate exactly -- math.modf to split the index, then
        Tensor.lerp between adjacent layers' directions, then L2 normalise. version_A could
        only read integer layers {16,20,24}; every winning Heretic trial sat at 12.6-14.2,
        a band it could not even express.

        Surgical uses the nearest integer layer's capability subspace: the subspace is a
        property of the layer's activations, not of the interpolated direction, and
        interpolating two SVD bases is not meaningful.
        """
        import math as _math
        frac, lo = _math.modf(float(layer))
        lo = int(lo)
        src = self.plain if cap_rank == 0 else None
        if src is None:
            key = (int(round(layer)), cap_rank)
            return self.surgical[key]
        hi = lo + 1
        if hi not in src:
            d = src[lo]
        else:
            d = src[lo].lerp(src[hi], frac)
        return d / d.norm().clamp(min=1e-9)

    def directions_for(self, spec: AttackSpec, rng: random.Random | None = None):
        """Return {layer: d} for per-layer attacks, else the single shared d.

        Per-layer always indexes integer layers (each attacked layer uses its own). Only the
        SHARED direction can be fractional, and only version_B emits those.

        `spec.jitter_deg > 0` rotates every returned direction by a random angle in
        [0, jitter_deg] toward a random orthogonal direction -- see `_jitter`.
        """
        rank_key = (int(round(spec.read_layer)), spec.attack_rank)
        if rank_key in self.ranked:
            if spec.variant != "plain" or spec.per_layer:
                raise ValueError("rank-k is supported only for shared canonical attacks")
            R = self.ranked[rank_key]
            if spec.jitter_deg > 0:
                J = torch.stack([_jitter(row, spec.jitter_deg, rng) for row in R])
                R = torch.linalg.qr(J.T)[0].T[: spec.attack_rank]
            return R
        pick = (lambda L: self.plain[L]) if spec.variant == "plain" \
            else (lambda L: self.surgical[(L, spec.cap_rank)])
        def j(d):
            return _jitter(d, spec.jitter_deg, rng)
        if spec.per_layer:
            return {L: j(pick(L)) for L in spec.layers}
        if float(spec.read_layer) != int(spec.read_layer):
            return j(self.direction_at(spec.read_layer, spec.cap_rank))
        return j(pick(int(spec.read_layer)))

    def realized_overlap(self, spec: AttackSpec) -> float:
        """Log this per step -- it is the axis version_A exists to vary."""
        if (int(round(spec.read_layer)), spec.attack_rank) in self.ranked:
            return float("nan")
        if spec.variant == "plain":
            ls = spec.layers if spec.per_layer else [spec.read_layer]
            # round: version_B read_layers are fractional and the bank is keyed by int
            return sum(self.overlap.get((int(round(L)), 16), float("nan"))
                       for L in ls) / len(ls)
        return 0.0


_TENT_KEY = {"self_attn.o_proj": "attn.o_proj", "mlp.down_proj": "mlp.down_proj"}


def sample_attack_c(
    rng: random.Random,
    n_layers: int,
    *,
    buffer=None,
    cap_ranks: tuple[int, ...] = (4, 8, 16),
    p_canonical: float = 0.20,
    p_surgical: float = 0.15,
    p_buffer: float = 0.30,
    p_heretic_dir: float = 0.60,
) -> AttackSpec:
    """version_C. Independent per-projection tents + heretic's application and direction.

    THE MIX, and why each slice is there:

    - `p_canonical` 0.20 -- the plain rank-1 Arditi attack, our headline result. version_A
      draws this FIRST and short-circuits for the same reason: if it is merely one outcome
      among many it loses gradient mass and the published claim quietly rots.
    - `p_surgical` 0.15 -- capability-orthogonalised directions. This is v8's break and the
      thing version_A/B fixed; drop it and the fix regresses.
    - `p_buffer` 0.30 -- replay of real Heretic winners against recent weights, jittered.
      Falls back to a random tent when the buffer is empty (i.e. before the first refresh),
      so the mix degrades gracefully rather than silently training on nothing.
    - remainder ~0.35 -- random independent tents, for coverage around whatever Heretic found.

    `p_heretic_dir` applies only to the tent slices; canonical and surgical keep our own
    recipe so their established results stay comparable. It is 0.60 rather than 1.0 because
    an attacker using plain mean-difference directions is still a real attacker, and the
    step-0 sensitivity result (4% direction error = 12.5 points of harmful rate) says the
    model should not be allowed to key its collapse to one direction recipe.
    """
    if rng.random() < p_canonical:
        rp, wp = SCOPES["all"]
        return AttackSpec(rp, wp, list(range(n_layers)), None, False, "plain", 0,
                          float(n_layers // 2), "canonical:arditi",
                          "plain", "ours")

    if rng.random() < p_surgical / (1.0 - p_canonical):
        rp, wp = SCOPES["all"]
        lo = rng.randint(0, n_layers - 4)
        hi = rng.randint(min(lo + 3, n_layers - 1), n_layers - 1)
        layers = list(range(lo, hi + 1))
        k = rng.choice(list(cap_ranks))
        return AttackSpec(rp, wp, layers, None, rng.random() < 0.5, "surgical", k,
                          float(rng.randint(0, n_layers - 1)), f"surgical:k{k}",
                          "plain", "ours")

    spec = None
    if buffer is not None and len(buffer) and rng.random() < p_buffer / (1.0 - p_canonical - p_surgical):
        t = buffer.sample(rng)
        if t is not None:
            spec = heretic_spec(t, n_layers, tag=f"buffer:t{t.get('trial', '?')}")

    if spec is None:
        profiles = {}
        for proj in HERETIC_PROJ:
            profiles[proj] = Tent(
                max_weight=rng.uniform(0.5, 1.5),
                max_pos=rng.uniform(0.25 * (n_layers - 1), 0.95 * (n_layers - 1)),
                min_weight=rng.uniform(0.0, 0.9),
                min_dist=rng.uniform(1.0, 0.6 * n_layers),
            )
        layers, alphas = tent_alphas(profiles, n_layers)
        if not layers:                      # every tent fell outside the stack
            return sample_attack_c(rng, n_layers, buffer=buffer, cap_ranks=cap_ranks,
                                   p_canonical=p_canonical, p_surgical=p_surgical,
                                   p_buffer=p_buffer, p_heretic_dir=p_heretic_dir)
        spec = AttackSpec((), HERETIC_PROJ, layers, alphas, rng.random() < 0.5,
                          "plain", 0, rng.uniform(0.25 * (n_layers - 1), 0.95 * (n_layers - 1)),
                          "tent:random")

    recipe = "heretic" if rng.random() < p_heretic_dir else "ours"
    # heretic always uses its FULL row-normalised application; a naive attacker using the
    # same tent shape would use the plain one, so sample it rather than assuming.
    application = "full" if rng.random() < 0.75 else "plain"
    return spec._replace(application=application, dir_recipe=recipe,
                         tag=f"{spec.tag}:{application}:{recipe}")


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
    for estimator in RANK_ESTIMATORS:
        R = _refusal_subspaces(H[:25], H[25:], (1, 2, 4), estimator)
        assert R[1].shape == (1, d_model) and R[4].shape == (4, d_model)
        assert torch.allclose(R[2].abs(), R[4][:2].abs(), atol=1e-5)
        assert torch.allclose(R[4] @ R[4].T, torch.eye(4), atol=1e-5)
    R = _refusal_subspaces(H[:25], H[25:], (1, 2, 4), "svd")
    bank = DirectionBank({3: d}, {(3, 2): R[2]}, {}, {(3, 16): 0.0})
    rank2 = AttackSpec(*SCOPES["all"], [0, 1], None, False, "plain", 0, 3.0,
                       "canonical:arditi:k2", attack_rank=2)
    out = bank.directions_for(rank2._replace(jitter_deg=10), random.Random(1))
    assert out.shape == (2, d_model)
    assert torch.allclose(out @ out.T, torch.eye(2), atol=1e-5)

    # sampler: every spec well-formed, and the canonical arm keeps its mass
    rng = random.Random(0)
    n_layers, n = 28, 4000
    specs = [sample_attack(rng, n_layers) for _ in range(n)]
    for s in specs:
        assert s.layers and all(0 <= L < n_layers for L in s.layers)
        assert s.alphas is None or all(0.2 <= a <= 1.0 for a in s.alphas.values())
        assert s.variant in {"plain", "surgical"}
        assert (s.cap_rank == 0) == (s.variant == "plain")
        # sub-layer ablations under-trigger the collapse, so no window may be narrow
        assert len(s.layers) >= n_layers // 2, (s.tag, len(s.layers))
    # ---- version_B ----
    rngb = random.Random(0)
    bspecs = [sample_attack_b(rngb, n_layers) for _ in range(4000)]
    write_only = 0
    for s in bspecs:
        assert s.layers and all(0 <= L < n_layers for L in s.layers)
        assert s.read_proj or s.write_proj, "empty matrix subset"
        assert s.alphas is None or all(0.2 <= a <= 1.5 for a in s.alphas.values())
        assert 0 <= s.read_layer <= n_layers - 1
        if s.write_proj and not s.read_proj:
            write_only += 1
    canonical_ranks = {s.attack_rank for s in bspecs if s.tag == "canonical:arditi"}
    assert canonical_ranks == {1}, canonical_ranks
    assert all(s.attack_rank == 1 for s in bspecs)
    rngg = random.Random(0)
    gspecs = [sample_attack_b(
        rngg, n_layers, canonical_ranks=VERSION_G_ATTACK_RANKS,
        p_canonical=0.10, p_heretic=0.35, p_surgical=0.40,
    ) for _ in range(4000)]
    got = {s.attack_rank for s in gspecs if s.tag.startswith("canonical:rank_k:")}
    assert got == set(VERSION_G_ATTACK_RANKS), got
    assert all(s.attack_rank == 1 for s in gspecs if not s.tag.startswith("canonical:rank_k:"))
    rank_counts = {
        rank: sum(s.attack_rank == rank and s.tag.startswith("canonical:rank_k:")
                  for s in gspecs)
        for rank in VERSION_G_ATTACK_RANKS
    }
    assert 0.08 < sum(rank_counts.values()) / len(gspecs) < 0.12, rank_counts
    assert all(0.012 < count / len(gspecs) < 0.028 for count in rank_counts.values()), rank_counts
    heretic_frac = sum(s.tag.startswith("heretic:tent:") for s in gspecs) / len(gspecs)
    surgical_frac = sum(s.variant == "surgical" for s in gspecs) / len(gspecs)
    assert 0.32 < heretic_frac < 0.38, heretic_frac
    assert 0.19 < surgical_frac < 0.25, surgical_frac
    print(f"version_g_final: rank_k={sum(rank_counts.values()) / len(gspecs):.3f} "
          f"heretic={heretic_frac:.3f} surgical={surgical_frac:.3f} ranks={rank_counts}")
    frac_wo = write_only / len(bspecs)
    frac_frac = sum(float(s.read_layer) != int(s.read_layer) for s in bspecs) / len(bspecs)
    lo_dir = sum(s.read_layer < 16 for s in bspecs) / len(bspecs)
    hi_alpha = sum(s.alphas is not None and max(s.alphas.values()) > 1.0 for s in bspecs) / len(bspecs)
    # the three gaps version_B exists to close must actually appear
    assert frac_wo > 0.03, f"write-only too rare: {frac_wo}"
    assert lo_dir > 0.25, f"sub-16 direction layers too rare: {lo_dir}"
    assert hi_alpha > 0.15, f"alpha>1 too rare: {hi_alpha}"
    print(f"version_B: write_only={frac_wo:.3f} fractional_dir={frac_frac:.3f} "
          f"dir<16={lo_dir:.3f} alpha>1={hi_alpha:.3f} "
          f"variants={len({s.tag for s in bspecs})}")

    canon = sum(s.tag == "canonical:arditi" for s in specs) / n
    surg = sum(s.variant == "surgical" for s in specs) / n
    assert 0.17 < canon < 0.23, canon
    assert 0.29 < surg < 0.35, surg          # 0.8 * 0.4
    print(f"ok  canonical={canon:.3f}  surgical={surg:.3f}  "
          f"variants={len({s.tag for s in specs})}")


def _selftest_jitter() -> None:
    """jitter must hit the requested angle band, stay unit-norm, and be a NO-OP at 0."""
    import math as _m
    rng = random.Random(0)
    d = torch.randn(512)
    d /= d.norm()

    # 0 degrees: identical object, and crucially the rng stream is untouched --
    # that is what makes every pre-existing run reproduce bit-identically.
    before = rng.getstate()
    assert _jitter(d, 0.0, rng) is d
    assert rng.getstate() == before, "jitter(0) consumed randomness"

    for deg in (15.0, 41.0, 60.0):
        angs = []
        for _ in range(200):
            o = _jitter(d, deg, rng)
            assert abs(float(o.norm()) - 1.0) < 1e-5, "not unit norm"
            angs.append(_m.degrees(_m.acos(max(-1.0, min(1.0, float(o @ d))))))
        assert max(angs) <= deg + 1e-3, f"exceeded {deg}: {max(angs)}"
        assert max(angs) > deg * 0.8, f"never approached {deg}: {max(angs)}"
        assert min(angs) < deg * 0.2, f"never stayed near 0: {min(angs)}"

    # AttackSpec default keeps every existing constructor at 0.
    spec = AttackSpec((), HERETIC_PROJ, [0], None, False, "plain", 0, 0.0, "t")
    assert spec.jitter_deg == 0.0
    print("jitter selftest ok")


if __name__ == "__main__":
    _selfcheck()
    _selftest_jitter()
