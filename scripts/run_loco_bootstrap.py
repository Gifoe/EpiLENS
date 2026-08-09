#!/usr/bin/env python3
"""Compute center-stratified patient bootstrap intervals for LOCO results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epilens.analyses import loco_patient_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    table = loco_patient_bootstrap(pd.read_csv(args.patient_metrics), args.bootstraps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
