"""tamperforge — a pre-release procedure to entangle safety with capability.

See ROADMAP.md and THREAT_MODEL.md. The goal is to make cheap automated
uncensoring (abliteration) self-defeating, not to make a model un-finetunable.
"""

from .abliterate import (
    abliterate_adapter_out_inplace,
    abliterate_model_inplace,
    adapter_wout_directions,
    orthonormalize_directions,
    project_out_read,
    project_out_write,
)
from .adapter import GatedSafetyAdapter, SafetyAdapter, load_adapter, make_adapter_hook
from .directions import empirical_refusal_direction, sae_feature_directions
from .fold import fold_gated_adapter_into_ffn, verify_fold
from .model import load_model, load_sae, pick_device, capture_residual
from .safety import REFUSAL_PHRASES, is_refusal

__all__ = [
    "abliterate_model_inplace",
    "abliterate_adapter_out_inplace",
    "adapter_wout_directions",
    "orthonormalize_directions",
    "project_out_read",
    "project_out_write",
    "SafetyAdapter",
    "GatedSafetyAdapter",
    "load_adapter",
    "make_adapter_hook",
    "fold_gated_adapter_into_ffn",
    "verify_fold",
    "empirical_refusal_direction",
    "sae_feature_directions",
    "load_model",
    "load_sae",
    "pick_device",
    "capture_residual",
    "REFUSAL_PHRASES",
    "is_refusal",
]
