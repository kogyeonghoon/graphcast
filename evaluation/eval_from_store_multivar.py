#!/usr/bin/env python
"""
Compute CRPS for GenCast 0.25‑degree sparse‑evaluation outputs (multi-variable)
===============================================================================

This script:

1.  Finds every pair of files
        predictions_025deg_<YYYYMMDD_HHMM>.nc
        targets_025deg_<YYYYMMDD_HHMM>.nc
    in the directory provided with --data_dir.

2.  For each pair, loads the datasets with xarray.

3.  Computes CRPS specifically for a specified surface variable,
    averages it spatially to obtain one CRPS value per initialization
    time stamp and lead time.

4.  Provides comprehensive analysis including CRPS by lead time,
    similar to the ECMWF evaluation script.

5.  Writes detailed CSV reports and prints a comprehensive summary.

Supported variables:
- 2t: 2-meter temperature (2m_temperature)
- 10u: 10m u-component of wind (10m_u_component_of_wind)
- 10v: 10m v-component of wind (10m_v_component_of_wind)
- msl: Mean sea level pressure (mean_sea_level_pressure)
- tp: Total precipitation (total_precipitation)

The implementation re‑uses the exact CRPS definition that was
embedded in your evaluation script, so results will be identical.

-----------------------------------------------------------------
Usage
-----

$ python eval_from_store_multivar.py \
        --data_dir predictions_025deg_sparse_output \
        --variable 2t

$ python eval_from_store_multivar.py \
        --data_dir predictions_025deg_sparse_output \
        --variable 10u \
        --output_csv gencast_10u_results.csv

Optional flags:

    --output_csv   Name of the CSV file to write
                   (auto-generated if not provided)

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
# Variable mapping and descriptions
# -------------------------------------------------------------------

# Variable mapping: argument variable name -> GenCast variable name
VARIABLE_MAPPING = {
    '2t': '2m_temperature',           # 2-meter temperature
    't2m': '2m_temperature',          # Alternative name for 2-meter temperature
    '10u': '10m_u_component_of_wind', # 10m u-component of wind
    'u10': '10m_u_component_of_wind', # Alternative name
    '10v': '10m_v_component_of_wind', # 10m v-component of wind
    'v10': '10m_v_component_of_wind', # Alternative name
    'msl': 'mean_sea_level_pressure', # Mean sea level pressure
    'tp': 'total_precipitation',      # Total precipitation
}

# Variable descriptions for output
VARIABLE_DESCRIPTIONS = {
    '2t': '2-meter temperature',
    't2m': '2-meter temperature',
    '10u': '10m u-component of wind',
    'u10': '10m u-component of wind',
    '10v': '10m v-component of wind',
    'v10': '10m v-component of wind',
    'msl': 'Mean sea level pressure',
    'tp': 'Total precipitation',
}

# Units for each variable
VARIABLE_UNITS = {
    '2t': 'K',
    't2m': 'K',
    '10u': 'm/s',
    'u10': 'm/s',
    '10v': 'm/s',
    'v10': 'm/s',
    'msl': 'Pa',
    'tp': 'm',
}


# -------------------------------------------------------------------
# CRPS utilities (copied from the evaluation script)
# -------------------------------------------------------------------

def crps(targets, predictions, bias_corrected=True):
    """CRPS for a single variable (supports ensemble in 'sample' dim)."""
    if predictions.sizes.get("sample", 1) < 2:
        raise ValueError("predictions must have dim 'sample' with size ≥ 2")

    # term 1: |y − x| averaged over ensemble members
    mean_abs_err = np.abs(targets - predictions).mean(dim="sample", skipna=False)

    # term 2: |xᵢ − xⱼ| averaged over all member pairs
    preds2 = predictions.rename({"sample": "sample2"})
    num = predictions.sizes["sample"]
    denom = (num - 1) if bias_corrected else num
    mean_abs_diff = (
        np.abs(predictions - preds2)
        .mean(dim=["sample", "sample2"], skipna=False)
        * (num / denom)
    )

    return mean_abs_err - 0.5 * mean_abs_diff


def crps_variable_only(targets, predictions, variable):
    """CRPS for specified variable only → spatial mean, keeping time dimension."""
    
    # Get the GenCast variable name
    if variable not in VARIABLE_MAPPING:
        raise ValueError(f"Unknown variable '{variable}'. Supported: {list(VARIABLE_MAPPING.keys())}")
    
    gencast_var_name = VARIABLE_MAPPING[variable]
    
    # Check if variable exists in both datasets
    if gencast_var_name not in targets.data_vars:
        print(f"Available variables in targets: {list(targets.data_vars)}")
        raise ValueError(f"Variable '{gencast_var_name}' not found in targets")
    
    if gencast_var_name not in predictions.data_vars:
        print(f"Available variables in predictions: {list(predictions.data_vars)}")
        raise ValueError(f"Variable '{gencast_var_name}' not found in predictions")
    
    print(f"Using variable: {gencast_var_name}")
    
    # Compute CRPS for the specified variable
    var_crps = crps(targets[gencast_var_name], predictions[gencast_var_name])  # lat × lon × time
    
    # Average spatially (keep time dimension)
    spatial_dims = [d for d in var_crps.dims if d not in {"time"}]
    if spatial_dims:
        var_crps = var_crps.mean(dim=spatial_dims)  # time
    
    return var_crps


def extract_lead_times_from_time_coord(time_coord):
    """Extract lead times in hours from time coordinate."""
    if len(time_coord) == 1:
        # Single time step, assume lead time 0
        return [0]
    
    # Calculate lead times as hours from first time step
    base_time = time_coord[0]
    lead_times = []
    for t in time_coord:
        lead_hours = int((t - base_time) / np.timedelta64(1, 'h'))
        lead_times.append(lead_hours)
    
    return lead_times


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
        description="Compute CRPS from saved GenCast predictions/targets (multi-variable)")
    parser.add_argument("--data_dir", type=str,
                        default="predictions_025deg_sparse_output",
                        help="Folder containing *_predictions.nc and *_targets.nc")
    parser.add_argument("--variable", type=str, required=True,
                        choices=['2t', 't2m', '10u', 'u10', '10v', 'v10', 'msl', 'tp'],
                        help="Variable to evaluate (2t, 10u, 10v, msl, tp)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Filename for detailed CSV output (auto-generated if not provided)")
    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output_csv is None:
        args.output_csv = f"gencast_crps_{args.variable}_results.csv"

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_dir():
        raise SystemExit(f"Directory not found: {data_dir}")

    pairs = find_file_pairs(data_dir)
    if not pairs:
        raise SystemExit("No matching prediction/target file pairs found.")

    results = []
    print("GenCast CRPS Evaluation (Multi-Variable)")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Variable: {args.variable} ({VARIABLE_DESCRIPTIONS.get(args.variable, 'Unknown')})")
    print(f"GenCast variable: {VARIABLE_MAPPING[args.variable]}")
    print(f"Units: {VARIABLE_UNITS.get(args.variable, 'Unknown')}")
    print(f"Found {len(pairs)} file pairs")
    print("=" * 50)
    
    for i, (ts, pred_path, tgt_path) in enumerate(pairs):
        print(f"\n[{i+1:2d}/{len(pairs)}] Processing: {ts}")
        
        try:
            preds = xr.open_dataset(pred_path, decode_timedelta=True)
            tgts = xr.open_dataset(tgt_path, decode_timedelta=True)

            # Get CRPS for the specified variable (keeps time dimension)
            crps_var = crps_variable_only(tgts, preds, args.variable)
            
            # Extract lead times from time coordinate
            lead_times = extract_lead_times_from_time_coord(crps_var.time)
            
            # Parse initialization timestamp
            init_time = pd.to_datetime(ts, format='%Y%m%d_%H%M')
            
            # Store results for each lead time
            for lead_hours, crps_value in zip(lead_times, crps_var.values):
                valid_time = init_time + pd.Timedelta(hours=lead_hours)
                
                results.append({
                    "variable": args.variable,
                    "timestamp": ts,
                    "init_time": init_time,
                    "lead_hours": lead_hours,
                    "valid_time": valid_time,
                    f"crps_{args.variable}": float(crps_value)
                })
                
                print(f"  Lead {lead_hours:3d}h → valid {valid_time}: CRPS = {crps_value:.6f}")
            
        except Exception as e:
            print(f"  ERROR: {e} – skipped")
            continue

    if not results:
        raise SystemExit("No CRPS values calculated – nothing to save.")

    # Create DataFrame and analysis
    df = pd.DataFrame(results).sort_values(["init_time", "lead_hours"])
    crps_col = f"crps_{args.variable}"
    
    # Save detailed results
    df.to_csv(args.output_csv, index=False)

    # Analysis similar to ECMWF script
    print(f"\n" + "=" * 60)
    print(f"EVALUATION SUMMARY")
    print(f"=" * 60)
    print(f"Variable: {args.variable} ({VARIABLE_DESCRIPTIONS.get(args.variable, 'Unknown')})")
    print(f"Total evaluations: {len(results)}")
    print(f"Number of initialization times: {df['init_time'].nunique()}")
    print(f"Number of lead times: {df['lead_hours'].nunique()}")

    # Overall statistics
    all_crps = df[crps_col].values
    print(f"\nOverall CRPS Statistics:")
    print(f"  Mean: {np.mean(all_crps):.6f}")
    print(f"  Std:  {np.std(all_crps):.6f}")
    print(f"  Min:  {np.min(all_crps):.6f}")
    print(f"  Max:  {np.max(all_crps):.6f}")

    # Statistics by lead time
    print(f"\nCRPS by Lead Time:")
    lead_time_stats = df.groupby('lead_hours')[crps_col].agg(['mean', 'std', 'count'])
    for lead_hours, stats in lead_time_stats.iterrows():
        print(f"  {lead_hours:3d}h: {stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['count']})")

    # Statistics by initialization time
    print(f"\nCRPS by Initialization Time (mean across all lead times):")
    init_time_stats = df.groupby('init_time')[crps_col].agg(['mean', 'std', 'count'])
    for init_time, stats in init_time_stats.iterrows():
        print(f"  {init_time}: {stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['count']})")

    # Save lead time statistics
    lead_stats_file = args.output_csv.replace('.csv', '_leadtime_stats.csv')
    lead_time_stats.to_csv(lead_stats_file)
    
    print(f"\n✓ Detailed results saved to {args.output_csv}")
    print(f"✓ Lead time statistics saved to {lead_stats_file}")
    
    print(f"\n🎯 GenCast evaluation completed!")
    print(f"   Variable: {args.variable} ({VARIABLE_DESCRIPTIONS.get(args.variable, 'Unknown')})")
    print(f"   Mean CRPS across all forecasts: {np.mean(all_crps):.6f}")


if __name__ == "__main__":
    main()