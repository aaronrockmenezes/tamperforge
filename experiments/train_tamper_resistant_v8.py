#!/usr/bin/env python
"""Compatibility import for historical commands; use train_version_g_final.py."""
from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("train_version_g_final", run_name="__main__")
else:
    import train_version_g_final as _impl

    globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
