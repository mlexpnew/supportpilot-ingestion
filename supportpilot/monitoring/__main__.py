"""CLI entry point for supportpilot.monitoring module."""

from __future__ import annotations

import sys
from supportpilot.monitoring.snapshot_baseline import main

if __name__ == "__main__":
    sys.exit(main())
