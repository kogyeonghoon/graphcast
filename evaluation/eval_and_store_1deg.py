"""
GenCast 1 Degree Evaluation Script - Sparse Evaluation Schedule with Prediction Saving

Efficient version that evaluates only at 6:00 UTC every 15 days starting from January 1st.
Uses local ERA5 data at native 1 degree resolution and evaluates on specified year.
Saves both CRPS results and actual predictions for further analysis.
"""

import os
import dataclasses
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import argparse
import time
from pathlib import Path

import haiku as hk
import jax
from google.cloud import storage

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
import logging
logging.getLogger().setLevel(logging.ERROR)

from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning

# Import utility functions from training script
from gencast_train_utils import get_local_era5_for_ngcm, get_batch_contiguous

# Configuration
os.environ["LD_LIBRARY_PATH"] = ""

# Constants
EVAL_YEAR = 2019
ROLLOUT_DAYS = 15
ROLLOUT_STEPS = ROLLOUT_DAYS * 2  # 12-hour steps
NUM_ENSEMBLE_MEMBERS = 8
BASE_DIR = "/home/work/kaist-siml/era5-1979-merged/GC_ERA5"
EVAL_INTERVAL_DAYS = 15  # Evaluate every 15 days
EVAL_HOUR = 6  # Only evaluate at 6:00 UTC

# Google Cloud Storage setup (for model and stats only)
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "gencast/"

gencast_dev_dir = '/home/work/kaist-siml-ssd/manual_v0529/graphcast/checkpoints/gencast_20250730_180241/step_0473849.pkl'

def crps(targets, predictions, bias_corrected=True):
    """Calculate CRPS - from original demo."""
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

def calculate_mean_crps_across_variables(targets, predictions):
    """Calculate mean CRPS across all variables and take spatial mean."""
    crps_values = []
    
    for var_name in targets.data_vars:
        if var_name in predictions.data_vars:
            var_crps = crps(targets[var_name], predictions[var_name])
            # Take spatial mean, keeping time dimension
            spatial_dims = [dim for dim in var_crps.dims if dim not in ['time']]
            if spatial_dims:
                var_crps_spatial_mean = var_crps.mean(dim=spatial_dims)
            else:
                var_crps_spatial_mean = var_crps
            crps_values.append(var_crps_spatial_mean)
    
    if crps_values:
        # Average across variables
        all_crps = xr.concat(crps_values, dim='variable')
        mean_crps = all_crps.mean(dim='variable')
        return mean_crps
    return None

def construct_wrapped_gencast():
    """Construct wrapped GenCast predictor - from original demo."""
    predictor = gencast.GenCast(
        sampler_config=sampler_config,
        task_config=task_config,
        denoiser_architecture_config=denoiser_architecture_config,
        noise_config=noise_config,
        noise_encoder_config=noise_encoder_config,
    )

    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    predictor = nan_cleaning.NaNCleaner(
        predictor=predictor,
        reintroduce_nans=True,
        fill_value=min_by_level,
        var_to_clean='sea_surface_temperature',
    )

    return predictor

def save_predictions_and_targets(predictions, targets, init_timestamp, output_dir):
    """Save predictions and targets to NetCDF files."""
    try:
        # Convert timestamp to string for filename safety
        timestamp_str = init_timestamp.strftime("%Y%m%d_%H%M")
        
        # Save predictions
        pred_filename = f"predictions_1deg_{timestamp_str}.nc"
        pred_path = output_dir / pred_filename
        
        # Add metadata to predictions
        predictions.attrs.update({
            'description': 'GenCast 1 degree ensemble predictions',
            'model_resolution': '1 degrees',
            'initialization_time': str(init_timestamp),
            'ensemble_members': predictions.sizes.get('sample', 1),
            'forecast_days': ROLLOUT_DAYS,
            'evaluation_schedule': f'Sparse evaluation every {EVAL_INTERVAL_DAYS} days at {EVAL_HOUR:02d}:00 UTC',
            'created_at': str(pd.Timestamp.now())
        })
        
        predictions.to_netcdf(pred_path, engine='netcdf4')
        print(f"    Saved predictions to {pred_path}")
        
        # Save targets
        target_filename = f"targets_1deg_{timestamp_str}.nc"
        target_path = output_dir / target_filename
        
        # Add metadata to targets
        targets.attrs.update({
            'description': 'ERA5 verification targets at 1 degree resolution',
            'model_resolution': '1 degrees',
            'initialization_time': str(init_timestamp),
            'forecast_days': ROLLOUT_DAYS,
            'evaluation_schedule': f'Sparse evaluation every {EVAL_INTERVAL_DAYS} days at {EVAL_HOUR:02d}:00 UTC',
            'created_at': str(pd.Timestamp.now())
        })
        
        targets.to_netcdf(target_path, engine='netcdf4')
        print(f"    Saved targets to {target_path}")
        
        return True
        
    except Exception as e:
        print(f"    ERROR: Failed to save predictions/targets - {e}")
        return False

def generate_sparse_evaluation_timestamps(year, interval_days=15, hour=6):
    """Generate sparse evaluation timestamps every `interval_days` starting from January 1st at specified hour."""
    eval_timestamps = []
    
    # Start from January 1st at specified hour
    current_date = pd.Timestamp(f"{year}-01-01 {hour:02d}:00:00")
    end_date = pd.Timestamp(f"{year}-12-31 23:59:59")
    
    while current_date <= end_date:
        eval_timestamps.append(current_date)
        current_date += pd.Timedelta(days=interval_days)
    
    return eval_timestamps

import pickle
def load_ckpt(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt

def main():
    parser = argparse.ArgumentParser(description='GenCast 1 Degree Sparse Evaluation with Local ERA5 Data and Prediction Saving')
    parser.add_argument('--ensemble_members', type=int, default=8, 
                        help='Number of ensemble members (default: 8)')
    parser.add_argument('--year', type=int, default=2019, 
                        help='Year to evaluate (default: 2019)')
    parser.add_argument('--interval_days', type=int, default=15,
                        help='Interval in days between evaluations (default: 15)')
    parser.add_argument('--eval_hour', type=int, default=6,
                        help='Hour (UTC) to evaluate at (default: 6)')
    parser.add_argument('--output_dir', type=str, default='predictions_1deg_sparse_output', 
                        help='Directory to save predictions (default: predictions_1deg_sparse_output)')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Save predictions to files (default: True)')
    parser.add_argument('--model', type=str, default='public',
                        choices=['public', 'dev'],
                        help='Model name to use: "public" or "dev" (default: public)')
    args = parser.parse_args()
    
    global sampler_config, task_config, denoiser_architecture_config
    global noise_config, noise_encoder_config
    global diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level
    global params, state
    global NUM_ENSEMBLE_MEMBERS, EVAL_YEAR, EVAL_INTERVAL_DAYS, EVAL_HOUR
    
    # Update configuration from arguments
    NUM_ENSEMBLE_MEMBERS = args.ensemble_members
    EVAL_YEAR = args.year
    EVAL_INTERVAL_DAYS = args.interval_days
    EVAL_HOUR = args.eval_hour
    
    # Create output directory for predictions
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(os.path.join(args.output_dir), args.model)
    if args.save_predictions:
        output_dir.mkdir(exist_ok=True)
        print(f"Predictions will be saved to: {output_dir.absolute()}")
    
    print("Starting GenCast 1 Degree Sparse Evaluation with Prediction Saving")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model: 1 degree resolution (native, no downsampling)")
    print(f"  Ensemble members: {NUM_ENSEMBLE_MEMBERS}")
    print(f"  Evaluation year: {EVAL_YEAR}")
    print(f"  Evaluation schedule: Every {EVAL_INTERVAL_DAYS} days at {EVAL_HOUR:02d}:00 UTC")
    print(f"  Save predictions: {args.save_predictions}")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of devices: {len(jax.devices())}")
    print("=" * 80)

    # Load 1 degree model checkpoint
    params_file = "GenCast 1p0deg Mini <2019.npz"
    print(f"Loading 1 degree model: {params_file}")
    
    with gcs_bucket.blob(dir_prefix + f"params/{params_file}").open("rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
        
    if args.model == 'public':
        params = ckpt.params
    else:
        ckpt_dev = load_ckpt(gencast_dev_dir)
        params = ckpt_dev['params']
    
    state = {}
    task_config = ckpt.task_config
    sampler_config = ckpt.sampler_config
    noise_config = ckpt.noise_config
    noise_encoder_config = ckpt.noise_encoder_config
    
    # Replace attention mechanism to avoid Triton backend issues (from GitHub issue #105)
    splash_spt_cfg = ckpt.denoiser_architecture_config.sparse_transformer_config
    tbd_spt_cfg = dataclasses.replace(splash_spt_cfg, attention_type="triblockdiag_mha", mask_type="full")
    denoiser_architecture_config = dataclasses.replace(ckpt.denoiser_architecture_config, sparse_transformer_config=tbd_spt_cfg)
    

    # configs = {
    #     "task_cfg": task_config,
    #     "sampler_cfg": sampler_config,
    #     "noise_cfg": noise_config,
    #     "noise_enc_cfg": noise_encoder_config,
    #     "denoiser_arch_cfg": denoiser_architecture_config,
    # }
    # for name, cfg in configs.items():
    #     print(f"\n{name} configuration:")
    #     for k, v in list(cfg.items())[:5]:
    #         print(f"{k}: {v}")

    print("Model description:", ckpt.description)
    
    # Load normalization data (following original demo)
    print("Loading normalization statistics...")
    with gcs_bucket.blob(dir_prefix+"stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xr.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xr.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xr.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/min_by_level.nc").open("rb") as f:
        min_by_level = xr.load_dataset(f).compute()
    
    # Note: Using local ERA5 data at native 1 degree resolution
    print("Using local ERA5 data at native 1 degree resolution (no downsampling)")
    
    # Build jitted functions (following original demo)
    print("Building model functions...")
    
    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    
    # Create pmapped version for parallel execution
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")
    
    print(f"Number of devices: {len(jax.local_devices())}")
    
    # Generate sparse evaluation timestamps
    eval_timestamps = generate_sparse_evaluation_timestamps(
        EVAL_YEAR, 
        interval_days=EVAL_INTERVAL_DAYS, 
        hour=EVAL_HOUR
    )
    
    print(f"Generated {len(eval_timestamps)} sparse evaluation timestamps:")
    print(f"  First: {eval_timestamps[0]}")
    print(f"  Last: {eval_timestamps[-1]}")
    print(f"  Schedule: Every {EVAL_INTERVAL_DAYS} days at {EVAL_HOUR:02d}:00 UTC")
    
    # Estimate time savings
    total_year_timestamps = 365 * 2  # 2 per day (6:00 and 18:00)
    time_reduction = (total_year_timestamps - len(eval_timestamps)) / total_year_timestamps
    print(f"  Time reduction: ~{time_reduction:.1%} fewer evaluations than full year")
    
    # Storage for CRPS values and timing results
    crps_results = []
    timing_results = []
    prediction_files = []  # Track saved prediction files
    
    # Run evaluation
    print(f"\nRunning sparse evaluation on {len(eval_timestamps)} timestamps...")
    
    for i, init_timestamp in enumerate(eval_timestamps):
        print(f"\n[{i+1:3d}/{len(eval_timestamps)}] Evaluating: {init_timestamp}")
        
        # Time the entire evaluation for this timestamp
        eval_start = time.time()
        
        try:
            # We need the timestep before the initial timestep for GenCast (2 input timesteps)
            prev_timestamp = init_timestamp - pd.Timedelta(hours=12)
            end_timestamp = init_timestamp + pd.Timedelta(days=ROLLOUT_DAYS)
            
            # Create time range including the previous timestep
            time_range = pd.date_range(
                start=prev_timestamp, 
                end=end_timestamp, 
                freq="6h"
            )
            
            print(f"  Loading data from {prev_timestamp} to {end_timestamp}")
            print(f"  Time range length: {len(time_range)} timesteps")
            
            # Time data loading
            data_loading_start = time.time()
            
            # Load local ERA5 data using the same approach as training script
            try:
                # Load raw data
                raw_data = get_local_era5_for_ngcm(
                    time_range, base_dir=BASE_DIR, compute=True
                )
                print(f"  Raw data shape: {raw_data.dims}")
                
                # Use get_batch_contiguous to properly format the data like in training
                # We need 2 input timesteps + rollout timesteps, using 6-hour native time steps
                window_size = 2 + ROLLOUT_STEPS  # 2 inputs + 30 rollout steps = 32 total
                window = tuple(range(0, window_size * 2, 2))  # (0, 2, 4, ..., 62) for 6-hour steps
                
                eval_data = get_batch_contiguous(
                    raw_data, 
                    start_ix=0, 
                    batch_size=1, 
                    window=window,
                    downsample=1  # Use native resolution for 1 degree (no downsampling)
                )
                print(f"  Formatted data shape: {eval_data.dims}")
                
                data_loading_time = time.time() - data_loading_start
                print(f"  ⏱️  Data loading time: {data_loading_time:.2f} seconds")
                
            except Exception as e:
                print(f"  ERROR: Failed to load data - {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Extract inputs, targets, and forcings (following original demo logic)
            eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
                eval_data, 
                target_lead_times=slice("12h", f"{ROLLOUT_STEPS*12}h"),
                **task_config.__dict__
            )
            
            print(f"  Input shape: {eval_inputs.dims}")
            print(f"  Target shape: {eval_targets.dims}")
            print(f"  Forcings shape: {eval_forcings.dims}")
            
            # Generate ensemble (following original demo)
            rng = jax.random.PRNGKey(42 + i)  # Different seed for each evaluation
            rngs = np.stack([
                jax.random.fold_in(rng, j) for j in range(NUM_ENSEMBLE_MEMBERS)
            ], axis=0)
            
            print(f"  Running {NUM_ENSEMBLE_MEMBERS}-member ensemble rollout...")
            
            # Time rollout
            rollout_start = time.time()
            
            # Perform rollout (following original demo)
            chunks = []
            for chunk in rollout.chunked_prediction_generator_multiple_runs(
                predictor_fn=run_forward_pmap,
                rngs=rngs,
                inputs=eval_inputs,
                targets_template=eval_targets * np.nan,
                forcings=eval_forcings,
                num_steps_per_chunk=1,
                num_samples=NUM_ENSEMBLE_MEMBERS,
                pmap_devices=jax.local_devices()
            ):
                chunks.append(chunk)
            
            predictions = xr.combine_by_coords(chunks)
            
            rollout_time = time.time() - rollout_start
            print(f"  Prediction shape: {predictions.dims}")
            print(f"  ⏱️  Rollout time: {rollout_time:.2f} seconds")
            
            # Save predictions and targets if requested
            save_start = time.time()
            saved_successfully = False
            if args.save_predictions:
                saved_successfully = save_predictions_and_targets(
                    predictions, eval_targets, init_timestamp, output_dir
                )
                if saved_successfully:
                    timestamp_str = init_timestamp.strftime("%Y%m%d_%H%M")
                    prediction_files.append({
                        'timestamp': init_timestamp,
                        'predictions_file': f"predictions_1deg_{timestamp_str}.nc",
                        'targets_file': f"targets_1deg_{timestamp_str}.nc"
                    })
            
            save_time = time.time() - save_start
            if args.save_predictions:
                print(f"  ⏱️  File saving time: {save_time:.2f} seconds")
            
            # Calculate CRPS
            mean_crps = calculate_mean_crps_across_variables(eval_targets, predictions)
            
            if mean_crps is not None:
                # Take temporal mean to get single CRPS value for this initial timestamp
                crps_value = float(mean_crps.mean().values)
                
                # Calculate total evaluation time
                total_eval_time = time.time() - eval_start
                
                crps_results.append({
                    'timestamp': init_timestamp,
                    'crps': crps_value,
                    'predictions_saved': saved_successfully
                })
                
                timing_results.append({
                    'timestamp': init_timestamp,
                    'data_loading_time': data_loading_time,
                    'rollout_time': rollout_time,
                    'save_time': save_time if args.save_predictions else 0.0,
                    'total_time': total_eval_time
                })
                
                print(f"  CRPS: {crps_value:.6f}")
                print(f"  ⏱️  Total evaluation time: {total_eval_time:.2f} seconds")
            else:
                print("  ERROR: Could not calculate CRPS")
                
        except Exception as e:
            print(f"  ERROR: Failed to evaluate timestamp {init_timestamp} - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Results summary
    if crps_results:
        crps_values = [r['crps'] for r in crps_results]
        crps_array = np.array(crps_values)
        
        # Timing statistics
        data_loading_times = [r['data_loading_time'] for r in timing_results]
        rollout_times = [r['rollout_time'] for r in timing_results]
        save_times = [r['save_time'] for r in timing_results]
        total_times = [r['total_time'] for r in timing_results]
        
        print(f"\nSparse Evaluation Summary (1 Degree Model):")
        print(f"Evaluation schedule: Every {EVAL_INTERVAL_DAYS} days at {EVAL_HOUR:02d}:00 UTC")
        print(f"Total evaluations: {len(crps_results)}")
        print(f"Mean CRPS: {crps_array.mean():.6f}")
        print(f"Std CRPS: {crps_array.std():.6f}")
        print(f"Min CRPS: {crps_array.min():.6f}")
        print(f"Max CRPS: {crps_array.max():.6f}")
        
        if args.save_predictions:
            successful_saves = sum(1 for r in crps_results if r.get('predictions_saved', False))
            print(f"Predictions saved: {successful_saves}/{len(crps_results)} evaluations")
        
        print(f"\nTiming Summary:")
        print(f"Average data loading time: {np.mean(data_loading_times):.2f} ± {np.std(data_loading_times):.2f} seconds")
        print(f"Average rollout time: {np.mean(rollout_times):.2f} ± {np.std(rollout_times):.2f} seconds")
        if args.save_predictions:
            print(f"Average save time: {np.mean(save_times):.2f} ± {np.std(save_times):.2f} seconds")
        print(f"Average total evaluation time: {np.mean(total_times):.2f} ± {np.std(total_times):.2f} seconds")
        print(f"Total time for all evaluations: {np.sum(total_times):.2f} seconds ({np.sum(total_times)/3600:.2f} hours)")
        
        # Save results
        results_df = pd.DataFrame(crps_results)
        results_filename = f'crps_results_1deg_sparse_{EVAL_YEAR}_{EVAL_INTERVAL_DAYS}day_{EVAL_HOUR:02d}utc.csv'
        results_df.to_csv(results_filename, index=False)
        print(f"\nResults saved to {results_filename}")
        
        # Save timing results
        timing_df = pd.DataFrame(timing_results)
        timing_filename = f'timing_results_1deg_sparse_{EVAL_YEAR}_{EVAL_INTERVAL_DAYS}day_{EVAL_HOUR:02d}utc.csv'
        timing_df.to_csv(timing_filename, index=False)
        print(f"Timing results saved to {timing_filename}")
        
        # Save prediction file list
        if prediction_files:
            pred_files_df = pd.DataFrame(prediction_files)
            pred_files_filename = f'prediction_files_1deg_sparse_{EVAL_YEAR}_{EVAL_INTERVAL_DAYS}day_{EVAL_HOUR:02d}utc.csv'
            pred_files_df.to_csv(pred_files_filename, index=False)
            print(f"Prediction files list saved to {pred_files_filename}")
        
        # Create histogram
        try:
            plt.figure(figsize=(12, 8))
            plt.hist(crps_values, bins=min(20, max(5, len(crps_values)//2)), alpha=0.7, edgecolor='black')
            plt.xlabel('CRPS')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of CRPS Values - GenCast 1° Sparse Evaluation\n{EVAL_YEAR} Every {EVAL_INTERVAL_DAYS} days at {EVAL_HOUR:02d}:00 UTC (n={len(crps_values)})')
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Mean: {crps_array.mean():.6f}\nStd: {crps_array.std():.6f}\nMin: {crps_array.min():.6f}\nMax: {crps_array.max():.6f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            histogram_filename = f'crps_histogram_1deg_sparse_{EVAL_YEAR}_{EVAL_INTERVAL_DAYS}day_{EVAL_HOUR:02d}utc.png'
            plt.savefig(histogram_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Histogram saved to {histogram_filename}")
        except Exception as e:
            print(f"Could not create histogram: {e}")
        
        # Create time series plot
        try:
            plt.figure(figsize=(14, 6))
            dates = [r['timestamp'] for r in crps_results]
            plt.plot(dates, crps_values, 'o-', linewidth=2, markersize=6)
            plt.xlabel('Date')
            plt.ylabel('CRPS')
            plt.title(f'CRPS Time Series - GenCast 1° Sparse Evaluation\n{EVAL_YEAR} Every {EVAL_INTERVAL_DAYS} days at {EVAL_HOUR:02d}:00 UTC')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Add horizontal line for mean
            plt.axhline(y=crps_array.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {crps_array.mean():.6f}')
            plt.legend()
            
            timeseries_filename = f'crps_timeseries_1deg_sparse_{EVAL_YEAR}_{EVAL_INTERVAL_DAYS}day_{EVAL_HOUR:02d}utc.png'
            plt.savefig(timeseries_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Time series plot saved to {timeseries_filename}")
        except Exception as e:
            print(f"Could not create time series plot: {e}")
            
        print(f"\n📁 All outputs saved to current directory and {output_dir if args.save_predictions else 'N/A'}")
        print(f"🚀 Sparse evaluation completed: {len(crps_results)} evaluations vs {365*2} for full year")
        print(f"⏱️  Total evaluation time: {np.sum(total_times)/3600:.2f} hours")
        
    else:
        print("No successful evaluations completed.")

if __name__ == "__main__":
    main() 