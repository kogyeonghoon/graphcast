#!/usr/bin/env python
"""
Compute CRPS for GenCast 0.25-degree sparse-evaluation outputs (pressure-level variables)
=========================================================================================

This script mirrors `eval_from_store_multivar.py` but targets pressure-level variables.

It:

1.  Finds every pair of files
        predictions_025deg_<YYYYMMDD_HHMM>.nc
        targets_025deg_<YYYYMMDD_HHMM>.nc
    in the directory provided with --data_dir.

2.  For each pair, loads the datasets with xarray.

3.  Computes CRPS for the specified pressure-level variable, either at a selected
    pressure level (e.g., 500 hPa) or aggregated over all available levels.

4.  Averages spatially to obtain one CRPS value per initialization time, lead time,
    and optionally level.

5.  Writes detailed CSV reports and prints a concise summary.

Supported pressure-level variables (argument → canonical):
- z, gh, geopotential_height → geopotential_height (if only 'geopotential' exists, it
  will be converted to height by dividing by g0 = 9.80665 m/s^2)
- t, temp, temperature → temperature
- q, sh, specific_humidity → specific_humidity
- u, u_component_of_wind → u_component_of_wind
- v, v_component_of_wind → v_component_of_wind

---------------------------------------------------------------------------------
Usage
-----

$ python eval_from_store_pl_multivar.py \
        --data_dir predictions_025deg_sparse_output \
        --variable z --level 500

$ python eval_from_store_pl_multivar.py \
        --data_dir predictions_025deg_sparse_output \
        --variable t --aggregate_levels

Optional flags:

    --output_csv        Name of the CSV file to write (auto-generated if not provided)
    --level             Pressure level in hPa (e.g., 500). If omitted and
                        --aggregate_levels not set, levels will be aggregated.
    --aggregate_levels  If set, CRPS is averaged across all available levels
                        (ignored when --level is provided)
    --level_dim         Name of the level dimension if auto-detection fails
                        (default: auto-detect among ['level','isobaricInhPa','pressure_level'])

---------------------------------------------------------------------------------
Requirements
------------

conda install -c conda-forge xarray netcdf4 numpy pandas
"""

from pathlib import Path
import argparse
import re
from typing import Optional, List
import xarray as xr
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------
# Variable mapping and descriptions (pressure-level)
# ---------------------------------------------------------------------------------

VARIABLE_MAPPING_PL = {
    # Geopotential height
    'z': 'geopotential_height',
    'gh': 'geopotential_height',
    'geopotential_height': 'geopotential_height',
    # Temperature
    't': 'temperature',
    'temp': 'temperature',
    'temperature': 'temperature',
    # Specific humidity
    'q': 'specific_humidity',
    'sh': 'specific_humidity',
    'specific_humidity': 'specific_humidity',
    # U wind
    'u': 'u_component_of_wind',
    'u_component_of_wind': 'u_component_of_wind',
    # V wind
    'v': 'v_component_of_wind',
    'v_component_of_wind': 'v_component_of_wind',
}

VARIABLE_DESCRIPTIONS_PL = {
    'z': 'Geopotential height',
    'gh': 'Geopotential height',
    'geopotential_height': 'Geopotential height',
    't': 'Temperature',
    'temp': 'Temperature',
    'temperature': 'Temperature',
    'q': 'Specific humidity',
    'sh': 'Specific humidity',
    'specific_humidity': 'Specific humidity',
    'u': 'U-component of wind',
    'u_component_of_wind': 'U-component of wind',
    'v': 'V-component of wind',
    'v_component_of_wind': 'V-component of wind',
}

VARIABLE_UNITS_PL = {
    'geopotential_height': 'm',
    'temperature': 'K',
    'specific_humidity': 'kg/kg',
    'u_component_of_wind': 'm/s',
    'v_component_of_wind': 'm/s',
}


# ---------------------------------------------------------------------------------
# CRPS utilities (same as surface script)
# ---------------------------------------------------------------------------------

def crps(targets: xr.DataArray, predictions: xr.DataArray, bias_corrected: bool = True) -> xr.DataArray:
    """CRPS for a single variable (supports ensemble in 'sample' dim)."""
    if predictions.sizes.get("sample", 1) < 2:
        raise ValueError("predictions must have dim 'sample' with size ≥ 2")

    mean_abs_err = np.abs(targets - predictions).mean(dim="sample", skipna=False)
    preds2 = predictions.rename({"sample": "sample2"})
    num = predictions.sizes["sample"]
    denom = (num - 1) if bias_corrected else num
    mean_abs_diff = (
        np.abs(predictions - preds2)
        .mean(dim=["sample", "sample2"], skipna=False)
        * (num / denom)
    )

    return mean_abs_err - 0.5 * mean_abs_diff


def detect_level_dim(var_da: xr.DataArray, override: Optional[str] = None) -> Optional[str]:
    """Detect the level dimension name from a few common candidates."""
    if override is not None:
        if override in var_da.dims:
            return override
        raise ValueError(f"Provided level_dim '{override}' not found in variable dims: {var_da.dims}")

    candidates = ["level", "isobaricInhPa", "pressure_level"]
    for cand in candidates:
        if cand in var_da.dims:
            return cand
    return None


def ensure_geopotential_height(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """Return geopotential height DataArray. If dataset holds 'geopotential', convert to height.

    Uses g0 = 9.80665 m/s^2.
    """
    if var_name == 'geopotential_height':
        if 'geopotential_height' in ds.data_vars:
            return ds['geopotential_height']
        if 'geopotential' in ds.data_vars:
            g0 = 9.80665
            return ds['geopotential'] / g0
        # Sometimes variable could be named 'z'
        if 'z' in ds.data_vars:
            return ds['z']
        raise ValueError("Neither 'geopotential_height' nor 'geopotential'/'z' found in dataset")
    raise RuntimeError("ensure_geopotential_height called for non-geopotential variable")


def get_pl_variable(ds: xr.Dataset, canonical_name: str) -> xr.DataArray:
    """Fetch the pressure-level variable array from dataset.

    Handles geopotential height conversion when necessary.
    """
    if canonical_name == 'geopotential_height':
        return ensure_geopotential_height(ds, canonical_name)

    if canonical_name not in ds.data_vars:
        available = list(ds.data_vars)
        raise ValueError(f"Variable '{canonical_name}' not found. Available: {available}")
    return ds[canonical_name]


def crps_pl_variable(
    targets: xr.Dataset,
    predictions: xr.Dataset,
    variable: str,
    level_hpa: Optional[int] = None,
    level_dim_override: Optional[str] = None,
) -> xr.DataArray:
    """CRPS for specified pressure-level variable.

    - If level_hpa is provided, compute CRPS at that level only.
    - Otherwise, compute CRPS across all levels and average over levels.
    Result keeps the time dimension; if level is selected, it keeps time only; if not,
    level is averaged out.
    """
    if variable not in VARIABLE_MAPPING_PL:
        raise ValueError(
            f"Unknown variable '{variable}'. Supported: {list(VARIABLE_MAPPING_PL.keys())}"
        )

    canonical = VARIABLE_MAPPING_PL[variable]

    tgt_da = get_pl_variable(targets, canonical)
    pred_da = get_pl_variable(predictions, canonical)

    level_dim = detect_level_dim(tgt_da, level_dim_override)
    if level_dim is None:
        raise ValueError(
            "Could not detect level dimension. Provide --level_dim explicitly."
        )

    # If a specific level is requested, select it
    if level_hpa is not None:
        try:
            tgt_sel = tgt_da.sel({level_dim: level_hpa})
            pred_sel = pred_da.sel({level_dim: level_hpa})
        except Exception as e:
            raise ValueError(
                f"Failed to select level {level_hpa} on dim '{level_dim}': {e}"
            )
        var_crps = crps(tgt_sel, pred_sel)
    else:
        # Compute CRPS at each level then average over levels
        var_crps_all_levels = crps(tgt_da, pred_da)
        var_crps = var_crps_all_levels.mean(dim=level_dim)

    # Average spatially (keep time dimension)
    spatial_dims = [d for d in var_crps.dims if d not in {"time"}]
    if spatial_dims:
        var_crps = var_crps.mean(dim=spatial_dims)

    return var_crps


def extract_lead_times_from_time_coord(time_coord: xr.DataArray) -> List[int]:
    """Extract lead times in hours from time coordinate."""
    if len(time_coord) == 1:
        return [0]
    base_time = time_coord[0]
    lead_times: List[int] = []
    for t in time_coord:
        lead_hours = int((t - base_time) / np.timedelta64(1, 'h'))
        lead_times.append(lead_hours)
    return lead_times


# ---------------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------------

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
        description="Compute CRPS from saved GenCast predictions/targets (pressure-level variables)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="predictions_025deg_sparse_output",
        help="Folder containing *_predictions.nc and *_targets.nc",
    )
    parser.add_argument(
        "--variable",
        type=str,
        required=True,
        choices=sorted(list(VARIABLE_MAPPING_PL.keys())),
        help="Pressure-level variable to evaluate (z/gh, t, q, u, v)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Pressure level in hPa (e.g., 500). If omitted and --aggregate_levels not set, levels are aggregated.",
    )
    parser.add_argument(
        "--aggregate_levels",
        action="store_true",
        help="Average CRPS across all available levels (ignored if --level is provided)",
    )
    parser.add_argument(
        "--level_dim",
        type=str,
        default=None,
        help="Name of the pressure-level dimension if auto-detection fails",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Filename for detailed CSV output (auto-generated if not provided)",
    )
    args = parser.parse_args()

    if args.output_csv is None:
        suffix = f"{args.variable}"
        if args.level is not None:
            suffix += f"_{args.level}hPa"
        elif args.aggregate_levels:
            suffix += "_alllevels"
        args.output_csv = f"gencast_crps_pl_{suffix}_results.csv"

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_dir():
        raise SystemExit(f"Directory not found: {data_dir}")

    pairs = find_file_pairs(data_dir)
    if not pairs:
        raise SystemExit("No matching prediction/target file pairs found.")

    results = []
    print("GenCast CRPS Evaluation (Pressure-Level Variables)")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Variable: {args.variable} ({VARIABLE_DESCRIPTIONS_PL.get(args.variable, 'Unknown')})")
    canonical = VARIABLE_MAPPING_PL[args.variable]
    print(f"GenCast variable: {canonical}")
    print(f"Units: {VARIABLE_UNITS_PL.get(canonical, 'Unknown')}")
    if args.level is not None:
        print(f"Level: {args.level} hPa")
    else:
        print("Levels: aggregated over all")
    print(f"Found {len(pairs)} file pairs")
    print("=" * 60)

    for i, (ts, pred_path, tgt_path) in enumerate(pairs):
        print(f"\n[{i+1:2d}/{len(pairs)}] Processing: {ts}")

        try:
            preds = xr.open_dataset(pred_path, decode_timedelta=True)
            tgts = xr.open_dataset(tgt_path, decode_timedelta=True)

            # Compute CRPS for the specified variable
            crps_var = crps_pl_variable(
                tgts,
                preds,
                args.variable,
                level_hpa=args.level,
                level_dim_override=args.level_dim,
            )

            # Extract lead times from time coordinate
            lead_times = extract_lead_times_from_time_coord(crps_var.time)

            # Parse initialization timestamp
            init_time = pd.to_datetime(ts, format='%Y%m%d_%H%M')

            # Store results for each lead time
            for lead_hours, crps_value in zip(lead_times, crps_var.values):
                valid_time = init_time + pd.Timedelta(hours=lead_hours)

                record = {
                    "variable": args.variable,
                    "timestamp": ts,
                    "init_time": init_time,
                    "lead_hours": lead_hours,
                    "valid_time": valid_time,
                }
                if args.level is not None:
                    record["level_hpa"] = int(args.level)
                record[f"crps_{args.variable}"] = float(crps_value)
                results.append(record)

                level_str = f" @ {args.level}hPa" if args.level is not None else " (avg levels)"
                print(f"  Lead {lead_hours:3d}h → valid {valid_time}{level_str}: CRPS = {crps_value:.6f}")

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

    # Summary
    print(f"\n" + "=" * 60)
    print(f"EVALUATION SUMMARY")
    print(f"=" * 60)
    print(f"Variable: {args.variable} ({VARIABLE_DESCRIPTIONS_PL.get(args.variable, 'Unknown')})")
    if args.level is not None:
        print(f"Level: {args.level} hPa")
    else:
        print("Levels: aggregated over all")
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

    print(f"\n🎯 GenCast PL evaluation completed!")
    print(f"   Variable: {args.variable} ({VARIABLE_DESCRIPTIONS_PL.get(args.variable, 'Unknown')})")
    if args.level is not None:
        print(f"   Level: {args.level} hPa")
    print(f"   Mean CRPS across all forecasts: {np.mean(all_crps):.6f}")


if __name__ == "__main__":
    main()


