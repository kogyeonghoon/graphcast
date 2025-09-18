#!/usr/bin/env python
"""
Evaluate ECMWF TIGGE forecasts against ERA5 ground truth (multi-variable)
=========================================================================

This script:

1. Loads ECMWF TIGGE forecasts from GRIB file for a specified variable
2. For each forecast initialization time (00 UTC) and lead time,
   loads corresponding ERA5 ground truth data
3. Computes CRPS between ECMWF ensemble forecasts and ERA5 truth
4. Averages CRPS across all lead times and produces summary statistics

The ECMWF forecasts start at 00 UTC on sparse dates (every 15 days in 2019),
while GenCast forecasts started at 06 UTC. This script properly handles the
different verification times.

Supported variables:
- 2t: 2-meter temperature
- 10u: 10m u-component of wind  
- 10v: 10m v-component of wind
- msl: Mean sea level pressure
- tp: Total precipitation

Usage:
------
python eval_ecmwf_vs_era5_multivar.py --grib_file tigge_2t_2019_15d.grib2 --variable 2t
python eval_ecmwf_vs_era5_multivar.py --grib_file tigge_10u_2019_15d.grib2 --variable 10u

Requirements:
-------------
conda install -c conda-forge xarray netcdf4 numpy pandas cfgrib
"""

import os
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import warnings

# Import utility functions from training script
import sys
sys.path.append('..')
from gencast_train_utils import get_local_era5_for_ngcm

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
BASE_DIR = "/home/work/kaist-siml-ssd/era5-merged2/GC_ERA5"

# Variable mapping: ECMWF GRIB variable name -> ERA5 variable name
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

def crps(targets, predictions, bias_corrected=True):
    """Calculate CRPS - from original eval script."""
    if predictions.sizes.get("sample", 1) < 2:
        raise ValueError("predictions must have dim 'sample' with size at least 2.")
        
    sum_dims = ["sample", "sample2"]
    preds2 = predictions.rename({"sample": "sample2"})
    num_samps = predictions.sizes["sample"]
    num_samps2 = (num_samps - 1) if bias_corrected else num_samps
    
    mean_abs_diff = np.abs(predictions - preds2).sum(
        dim=sum_dims, skipna=False) / (num_samps * num_samps2)
    mean_abs_err = np.abs(targets - predictions).sum(
        dim="sample", skipna=False) / num_samps
    
    return mean_abs_err - 0.5 * mean_abs_diff

def load_era5_for_verification(valid_time, variable, base_dir=BASE_DIR):
    """
    Load ERA5 data for a specific verification time and variable.
    
    Args:
        valid_time: pandas Timestamp for the verification time
        variable: ECMWF variable name (e.g., '2t', '10u', etc.)
        base_dir: Base directory for ERA5 data
    
    Returns:
        ERA5 dataset with the specified variable at the verification time
    """
    # Get ERA5 variable name
    if variable not in VARIABLE_MAPPING:
        raise ValueError(f"Unknown variable '{variable}'. Supported: {list(VARIABLE_MAPPING.keys())}")
    
    era5_var_name = VARIABLE_MAPPING[variable]
    
    # Create a time range with just the verification time
    time_range = pd.date_range(start=valid_time, end=valid_time, freq="6h")
    
    try:
        # Load ERA5 data using the same function as in evaluation scripts
        era5_data = get_local_era5_for_ngcm(
            time_range, base_dir=base_dir, compute=True
        )
        
        # Extract only the specified variable and ensure it has the right structure
        if era5_var_name in era5_data.data_vars:
            era5_var = era5_data[era5_var_name]
        else:
            raise ValueError(f"Variable '{era5_var_name}' not found in ERA5 data. Available: {list(era5_data.data_vars)}")
        
        return era5_var
        
    except Exception as e:
        print(f"    ERROR: Failed to load ERA5 data for {valid_time}, variable '{variable}' - {e}")
        return None

def compute_crps_for_lead_time(ecmwf_ensemble, era5_truth):
    """
    Compute CRPS between ECMWF ensemble and ERA5 truth for a specific lead time.
    
    Args:
        ecmwf_ensemble: xarray DataArray with ensemble dimension
        era5_truth: xarray DataArray with truth data
    
    Returns:
        CRPS value (spatially averaged)
    """
    try:
        # Ensure both datasets are on the same grid
        # ERA5 is at 0.25° resolution, ECMWF TIGGE should be too
        
        # Rename number dimension to sample for CRPS calculation
        if 'number' in ecmwf_ensemble.dims:
            ecmwf_ensemble = ecmwf_ensemble.rename({'number': 'sample'})
        
        # Ensure ERA5 truth matches the spatial grid
        # We might need to interpolate if grids don't exactly match
        if not (np.allclose(ecmwf_ensemble.latitude, era5_truth.latitude, atol=1e-5) and 
                np.allclose(ecmwf_ensemble.longitude, era5_truth.longitude, atol=1e-5)):
            print("    Interpolating ERA5 to ECMWF grid...")
            era5_truth = era5_truth.interp(
                latitude=ecmwf_ensemble.latitude,
                longitude=ecmwf_ensemble.longitude,
                method='linear'
            )
        
        # Compute CRPS
        crps_field = crps(era5_truth, ecmwf_ensemble)
        
        # Average spatially to get single CRPS value
        spatial_dims = [dim for dim in crps_field.dims if dim in ['latitude', 'longitude']]
        if spatial_dims:
            crps_value = float(crps_field.mean(dim=spatial_dims).values)
        else:
            crps_value = float(crps_field.values)
        
        return crps_value
        
    except Exception as e:
        print(f"    ERROR: Failed to compute CRPS - {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ECMWF TIGGE forecasts against ERA5 ground truth (multi-variable)'
    )
    parser.add_argument('--grib_file', type=str, required=True,
                        help='Path to TIGGE GRIB file')
    parser.add_argument('--variable', type=str, required=True,
                        choices=['2t', 't2m', '10u', 'u10', '10v', 'v10', 'msl', 'tp'],
                        help='Variable to evaluate (2t, 10u, 10v, msl, tp)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV file for detailed results (auto-generated if not provided)')
    parser.add_argument('--max_lead_hours', type=int, default=360,
                        help='Maximum lead time in hours to evaluate (default: 360)')
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output_csv is None:
        grib_basename = Path(args.grib_file).stem
        args.output_csv = f'ecmwf_vs_era5_{args.variable}_{grib_basename}_results.csv'
    
    print("ECMWF TIGGE vs ERA5 Evaluation (Multi-Variable)")
    print("=" * 60)
    print(f"GRIB file: {args.grib_file}")
    print(f"Variable: {args.variable} ({VARIABLE_DESCRIPTIONS.get(args.variable, 'Unknown')})")
    print(f"ERA5 variable: {VARIABLE_MAPPING[args.variable]}")
    print(f"ERA5 data directory: {BASE_DIR}")
    print(f"Maximum lead time: {args.max_lead_hours} hours")
    
    # Load ECMWF TIGGE data
    print(f"\nLoading ECMWF TIGGE data from {args.grib_file}...")
    try:
        ecmwf_data = xr.open_dataset(args.grib_file, engine='cfgrib')
        print(f"✓ Loaded ECMWF data")
        print(f"  Variables: {list(ecmwf_data.data_vars)}")
        print(f"  Dimensions: {dict(ecmwf_data.dims)}")
        print(f"  Initialization times: {len(ecmwf_data.time)} dates")
        print(f"  Ensemble members: {len(ecmwf_data.number)}")
        print(f"  Forecast steps: {len(ecmwf_data.step)}")
    except Exception as e:
        print(f"✗ Failed to load ECMWF data: {e}")
        return
    
    # Get the target variable data
    # Try both the requested variable name and common alternatives
    possible_var_names = [args.variable]
    if args.variable == '2t':
        possible_var_names.append('t2m')
    elif args.variable == 't2m':
        possible_var_names.append('2t')
    elif args.variable == '10u':
        possible_var_names.append('u10')
    elif args.variable == 'u10':
        possible_var_names.append('10u')
    elif args.variable == '10v':
        possible_var_names.append('v10')
    elif args.variable == 'v10':
        possible_var_names.append('10v')
    
    var_data = None
    found_var_name = None
    for var_name in possible_var_names:
        if var_name in ecmwf_data.data_vars:
            var_data = ecmwf_data[var_name]
            found_var_name = var_name
            break
    
    if var_data is None:
        print(f"✗ Variable '{args.variable}' not found. Available variables: {list(ecmwf_data.data_vars)}")
        print(f"  Tried: {possible_var_names}")
        return
    
    print(f"  Using variable: {found_var_name}")
    
    # Filter steps to maximum lead time
    max_lead_timedelta = pd.Timedelta(hours=args.max_lead_hours)
    valid_steps = var_data.step <= max_lead_timedelta
    var_data = var_data.sel(step=valid_steps)
    
    print(f"  Using {len(var_data.step)} forecast steps (up to {args.max_lead_hours}h)")
    
    # Storage for results
    results = []
    timing_results = []
    
    total_evaluations = len(var_data.time) * len(var_data.step)
    print(f"\nEvaluating {total_evaluations} forecast cases...")
    
    eval_count = 0
    for i, init_time in enumerate(var_data.time):
        init_timestamp = pd.Timestamp(init_time.values)
        print(f"\n[{i+1:2d}/{len(var_data.time)}] Init: {init_timestamp}")
        
        for j, step in enumerate(var_data.step):
            eval_count += 1
            step_hours = int(step.values / np.timedelta64(1, 'h'))
            
            # Calculate verification time
            valid_time = init_timestamp + pd.Timedelta(hours=step_hours)
            
            print(f"  [{eval_count:3d}/{total_evaluations}] Step {step_hours:3d}h → verify at {valid_time}")
            
            eval_start = time.time()
            
            try:
                # Extract ECMWF ensemble for this init time and step
                ecmwf_forecast = var_data.sel(time=init_time, step=step)
                
                # Load ERA5 ground truth for verification time
                era5_truth = load_era5_for_verification(valid_time, args.variable)
                if era5_truth is None:
                    continue
                
                # Remove time dimension from ERA5 if present (should be single timestep)
                if 'time' in era5_truth.dims:
                    era5_truth = era5_truth.squeeze('time')
                
                # Compute CRPS
                crps_value = compute_crps_for_lead_time(ecmwf_forecast, era5_truth)
                
                if crps_value is not None:
                    eval_time = time.time() - eval_start
                    
                    results.append({
                        'variable': args.variable,
                        'init_time': init_timestamp,
                        'lead_hours': step_hours,
                        'valid_time': valid_time,
                        'crps': crps_value
                    })
                    
                    timing_results.append({
                        'init_time': init_timestamp,
                        'lead_hours': step_hours,
                        'eval_time': eval_time
                    })
                    
                    print(f"    CRPS: {crps_value:.6f} (⏱️ {eval_time:.2f}s)")
                else:
                    print(f"    ✗ CRPS computation failed")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
    
    # Results analysis
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\n" + "=" * 60)
        print(f"EVALUATION SUMMARY")
        print(f"=" * 60)
        print(f"Variable: {args.variable} ({VARIABLE_DESCRIPTIONS.get(args.variable, 'Unknown')})")
        print(f"Total successful evaluations: {len(results)}")
        print(f"Total possible evaluations: {total_evaluations}")
        print(f"Success rate: {len(results)/total_evaluations:.1%}")
        
        # Overall statistics
        all_crps = results_df['crps'].values
        print(f"\nOverall CRPS Statistics:")
        print(f"  Mean: {np.mean(all_crps):.6f}")
        print(f"  Std:  {np.std(all_crps):.6f}")
        print(f"  Min:  {np.min(all_crps):.6f}")
        print(f"  Max:  {np.max(all_crps):.6f}")
        
        # Statistics by lead time
        print(f"\nCRPS by Lead Time:")
        lead_time_stats = results_df.groupby('lead_hours')['crps'].agg(['mean', 'std', 'count'])
        for lead_hours, stats in lead_time_stats.iterrows():
            print(f"  {lead_hours:3d}h: {stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['count']})")
        
        # Statistics by initialization time
        print(f"\nCRPS by Initialization Time (mean across all lead times):")
        init_time_stats = results_df.groupby('init_time')['crps'].agg(['mean', 'std', 'count'])
        for init_time, stats in init_time_stats.iterrows():
            print(f"  {init_time}: {stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['count']})")
        
        # Timing statistics
        if timing_results:
            timing_df = pd.DataFrame(timing_results)
            eval_times = timing_df['eval_time'].values
            print(f"\nTiming Statistics:")
            print(f"  Mean evaluation time: {np.mean(eval_times):.2f} ± {np.std(eval_times):.2f} seconds")
            print(f"  Total evaluation time: {np.sum(eval_times):.1f} seconds ({np.sum(eval_times)/60:.1f} minutes)")
        
        # Save results
        results_df.to_csv(args.output_csv, index=False)
        print(f"\n✓ Detailed results saved to {args.output_csv}")
        
        # Save lead time statistics
        lead_stats_file = args.output_csv.replace('.csv', '_leadtime_stats.csv')
        lead_time_stats.to_csv(lead_stats_file)
        print(f"✓ Lead time statistics saved to {lead_stats_file}")
        
        print(f"\n🎯 ECMWF TIGGE evaluation completed!")
        print(f"   Variable: {args.variable} ({VARIABLE_DESCRIPTIONS.get(args.variable, 'Unknown')})")
        print(f"   Mean CRPS across all forecasts: {np.mean(all_crps):.6f}")
        
    else:
        print(f"\n✗ No successful evaluations completed.")

if __name__ == "__main__":
    main()