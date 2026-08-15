"""tamperforge — a pre-release procedure to entangle safety with capability.

See ROADMAP.md and THREAT_MODEL.md. The goal is to make cheap automated
uncensoring (abliteration) self-defeating, not to make a model un-finetunable.
"""

from .abliterate import (
    abliteration_parameter_layout,
    abliteration_modules,
    abliterate_adapter_out_inplace,
    abliterate_model_inplace,
    adapter_wout_directions,
    orthonormalize_directions,
    project_out_read,
    project_out_write,
)
from .adapter import GatedSafetyAdapter, SafetyAdapter, load_adapter, make_adapter_hook
from .directions import (
    RANK_ESTIMATORS,
    capability_subspace_from_activations,
    empirical_refusal_direction,
    empirical_refusal_directions,
    refusal_subspaces_from_activations,
    sae_feature_directions,
    svd_refusal_directions,
    surgicalize_refusal_subspace,
)
from .fold import fold_gated_adapter_into_ffn, verify_fold
from .model import (
    apply_chat_template_no_think,
    capture_residual,
    capture_residuals,
    decoder_layers,
    load_model,
    load_partial_checkpoint,
    load_sae,
    pick_device,
    tokenize_chat_prompts,
)
from .safety import REFUSAL_PHRASES, is_refusal

__all__ = [
    "abliterate_model_inplace",
    "abliteration_parameter_layout",
    "abliteration_modules",
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
    "empirical_refusal_directions",
    "RANK_ESTIMATORS",
    "refusal_subspaces_from_activations",
    "capability_subspace_from_activations",
    "surgicalize_refusal_subspace",
    "sae_feature_directions",
    "svd_refusal_directions",
    "load_model",
    "load_partial_checkpoint",
    "load_sae",
    "pick_device",
    "capture_residual",
    "capture_residuals",
    "decoder_layers",
    "apply_chat_template_no_think",
    "tokenize_chat_prompts",
    "REFUSAL_PHRASES",
    "is_refusal",
]
