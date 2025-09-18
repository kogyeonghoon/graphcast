#!/usr/bin/env python
"""
Compute CRPS for GenCast 0.25‑degree sparse‑evaluation outputs
=============================================================

This script:

1.  Finds every pair of files
        predictions_025deg_<YYYYMMDD_HHMM>.nc
        targets_025deg_<YYYYMMDD_HHMM>.nc
    in the directory provided with --data_dir.

2.  For each pair, loads the datasets with xarray.

3.  Computes CRPS for every variable, averages it spatially,
    and then averages across variables to obtain one CRPS value
    per initialization time stamp.

4.  Writes a CSV report (`crps_results.csv`) and prints a
    concise summary to stdout.

The implementation re‑uses the exact CRPS definition that was
embedded in your evaluation script, so results will be identical.

-----------------------------------------------------------------
Usage
-----

$ python crps_from_saved_predictions.py \
        --data_dir predictions_025deg_sparse_output

Optional flags:

    --output_csv   Name of the CSV file to write
                   (default: crps_results.csv)

-----------------------------------------------------------------
Requirements
------------

conda install -c conda-forge xarray netcdf4 numpy pandas
"""

from pathlib import Path
import argparse
import re
import xarray as xr
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# CRPS utilities (copied verbatim from the evaluation script)
# -------------------------------------------------------------------

def crps(targets, predictions, bias_corrected=True):
    """CRPS for a single variable (supports ensemble in 'sample' dim)."""
    if predictions.sizes.get("sample", 1) < 2:
        raise ValueError("predictions must have dim 'sample' with size ≥ 2")

    # term 1: |y − x| averaged over ensemble members
    mean_abs_err = np.abs(targets - predictions).mean(dim="sample", skipna=False)

    # term 2: |xᵢ − xⱼ| averaged over all member pairs
    preds2 = predictions.rename({"sample": "sample2"})
    num = predictions.sizes["sample"]
    denom = (num - 1) if bias_corrected else num
    mean_abs_diff = (
        np.abs(predictions - preds2)
        .mean(dim=["sample", "sample2"], skipna=False)
        * (num / denom)
    )

    return mean_abs_err - 0.5 * mean_abs_diff


def mean_crps_across_vars(targets, predictions):
    """Variable‑wise CRPS → spatial mean → average over variables."""
    crps_values = []
    for var in targets.data_vars:
        if var not in predictions.data_vars:
            continue
        var_crps = crps(targets[var], predictions[var])           #  lat × lon × time
        spatial_dims = [d for d in var_crps.dims if d not in {"time"}]
        if spatial_dims:
            var_crps = var_crps.mean(dim=spatial_dims)            #  time
        crps_values.append(var_crps)

    if not crps_values:
        return None

    all_crps = xr.concat(crps_values, dim="variable")             # variable × time
    return all_crps.mean(dim="variable")                          # time


# -------------------------------------------------------------------
# Main routine
# -------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{4})\.nc$")


def find_file_pairs(data_dir: Path):
    """Return list of (timestamp, pred_file, tgt_file) tuples."""
    preds = {TIMESTAMP_RE.search(p.name).group(1): p
             for p in data_dir.glob("predictions_025deg_*.nc")}
    tgts = {TIMESTAMP_RE.search(p.name).group(1): p
            for p in data_dir.glob("targets_025deg_*.nc")}

    common = sorted(set(preds) & set(tgts))
    return [(ts, preds[ts], tgts[ts]) for ts in common]


def main():
    parser = argparse.ArgumentParser(
        description="Compute CRPS from saved GenCast predictions/targets")
    parser.add_argument("--data_dir", type=str,
                        default="predictions_025deg_sparse_output",
                        help="Folder containing *_predictions.nc and *_targets.nc")
    parser.add_argument("--output_csv", type=str,
                        default="crps_results.csv",
                        help="Filename for CSV output")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_dir():
        raise SystemExit(f"Directory not found: {data_dir}")

    pairs = find_file_pairs(data_dir)
    if not pairs:
        raise SystemExit("No matching prediction/target file pairs found.")

    results = []
    print(f"Found {len(pairs)} file pairs – computing CRPS ...")
    for ts, pred_path, tgt_path in pairs:
        preds = xr.open_dataset(pred_path)
        tgts = xr.open_dataset(tgt_path)

        mean_crps = mean_crps_across_vars(tgts, preds)
        if mean_crps is None:
            print(f"  {ts}: no overlapping variables – skipped")
            continue

        crps_value = float(mean_crps.mean().values)
        results.append({"timestamp": ts, "crps": crps_value})
        print(f"  {ts}: CRPS = {crps_value:.6f}")

    if not results:
        raise SystemExit("No CRPS values calculated – nothing to save.")

    df = pd.DataFrame(results).sort_values("timestamp")
    df.to_csv(args.output_csv, index=False)

    print("\nSummary")
    print("-------")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Mean  CRPS: {df['crps'].mean():.6f}")
    print(f"Std   CRPS: {df['crps'].std():.6f}")
    print(f"Min   CRPS: {df['crps'].min():.6f}")
    print(f"Max   CRPS: {df['crps'].max():.6f}")


if __name__ == "__main__":
    main()
