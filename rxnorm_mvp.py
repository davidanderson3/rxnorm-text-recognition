#!/usr/bin/env python3
"""Backward-compatible entrypoint for RxNorm text recognition."""

from rxnorm_text_recognition import *  # noqa: F401,F403
from rxnorm_text_recognition import main


if __name__ == "__main__":
    raise SystemExit(main())
