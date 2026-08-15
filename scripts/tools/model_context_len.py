#!/usr/bin/env python
"""Print a requested context length capped to the model config's declared maximum."""

import sys

from transformers import AutoConfig


def main() -> None:
    model, requested = sys.argv[1], int(sys.argv[2])
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    config = getattr(config, "text_config", config)
    limits = [
        getattr(config, key, None)
        for key in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length")
    ]
    limits = [int(value) for value in limits if isinstance(value, (int, float)) and 0 < value < 1_000_000]
    print(min([requested] + limits))


if __name__ == "__main__":
    main()
