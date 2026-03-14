#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys

import torch


def main() -> int:
    flash_attn_spec = importlib.util.find_spec("flash_attn")
    if flash_attn_spec is None:
        print("flash_attn import: missing")
        return 1

    print("flash_attn import: ok")
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"device count: {torch.cuda.device_count()}")
        print(f"device 0: {torch.cuda.get_device_name(0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
