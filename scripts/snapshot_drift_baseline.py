#!/usr/bin/env python3
"""CLI script to snapshot predicted-label mix from production for drift monitoring (SP-120).

Usage:
    python scripts/snapshot_drift_baseline.py [--output supportpilot/monitoring/drift_baseline.json]
"""

from __future__ import annotations

import sys
from supportpilot.monitoring.snapshot_baseline import main

if __name__ == "__main__":
    sys.exit(main())
