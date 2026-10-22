#!/usr/bin/env python3
"""Native-GPU entry point for the resumable Adaptive SEM protocol.

This thin entry point intentionally delegates to ``run_pipeline.py`` so the
command shown in the README remains stable while the orchestration code stays
in one testable module.
"""

from __future__ import annotations

from run_pipeline import main


if __name__ == "__main__":
    main()
