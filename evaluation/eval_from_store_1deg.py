from pathlib import Path
import argparse
import re
import xarray as xr
import numpy as np
import pandas as pd


# def main():


# -------------------------------------------------------------------
# Main routine
# -------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{4})\.nc$")


def find_file_pairs(data_dir: Path):
    """Return list of (timestamp, pred_file, tgt_file) tuples."""
    preds = {TIMESTAMP_RE.search(p.name).group(1): p
             for p in data_dir.glob("predictions_1deg_*.nc")}
    tgts = {TIMESTAMP_RE.search(p.name).group(1): p
            for p in data_dir.glob("targets_1deg_*.nc")}

    common = sorted(set(preds) & set(tgts))
    return [(ts, preds[ts], tgts[ts]) for ts in common]


import os
parser = argparse.ArgumentParser(
    description="Compute CRPS from saved GenCast predictions/targets")
parser.add_argument("--data_dir", type=str,
                    default="predictions_1deg_sparse_output/dev",
                    help="Folder containing *_predictions.nc and *_targets.nc")
# parser.add_argument("--output_csv", type=str,
#                     default="crps_results.csv",
#                     help="Filename for CSV output")
args = parser.parse_args()
# args.data_dir = '../predictions_1deg_sparse_output/dev'
# output_csv = os.path.join(args.data_dir, args.output_csv)

data_dir = Path(args.data_dir).expanduser()
if not data_dir.is_dir():
    raise SystemExit(f"Directory not found: {data_dir}")

pairs = find_file_pairs(data_dir)
if not pairs:
    raise SystemExit("No matching prediction/target file pairs found.")



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
def rmse(targets, predictions):
    """
    Root-mean-square error between targets and the **ensemble mean**
    (average over 'sample' dim).  No spatial averaging yet.
    """
    if "sample" not in predictions.dims:
        raise ValueError("predictions must have a 'sample' dimension")
    ens_mean = predictions.mean(dim="sample", skipna=False)
    return np.sqrt((targets - ens_mean) ** 2)                  # keep all dims


# ------------------------------------------------------------------
#  Collect CRPS **and** RMSE for every (var, level, rollout_time)
# ------------------------------------------------------------------
def metrics_by_var_level_time(tgts: xr.Dataset, preds: xr.Dataset) -> list[dict]:
    records = []

    for var in tgts.data_vars:
        if var not in preds.data_vars:
            continue

        var_crps  = crps(tgts[var], preds[var])                # retains all dims
        var_rmse  = rmse(tgts[var], preds[var])                # idem

        # spatial mean (batch, lat, lon) – keep time & level
        for dim in {"batch", "lat", "lon"} & set(var_crps.dims):
            var_crps = var_crps.mean(dim=dim)
            var_rmse = var_rmse.mean(dim=dim)

        if "level" in var_crps.dims:                           # 4-D variables
            for lev in var_crps["level"].values:
                sel_crps = var_crps.sel(level=lev)
                sel_rmse = var_rmse.sel(level=lev)
                for t in sel_crps["time"].values:
                    records.append(
                        dict(
                            variable=var,
                            level=int(lev),
                            rollout_time=pd.Timedelta(t).isoformat(),
                            crps=float(sel_crps.sel(time=t).values),
                            rmse=float(sel_rmse.sel(time=t).values),
                        )
                    )
        else:                                                  # 3-D variables
            for t in var_crps["time"].values:
                records.append(
                    dict(
                        variable=var,
                        level=None,
                        rollout_time=pd.Timedelta(t).isoformat(),
                        crps=float(var_crps.sel(time=t).values),
                        rmse=float(var_rmse.sel(time=t).values),
                    )
                )
    return records


# results = []
print(f"Found {len(pairs)} file pairs – computing CRPS ...")
for ts, pred_path, tgt_path in pairs:
    preds = xr.open_dataset(pred_path)
    tgts = xr.open_dataset(tgt_path)

    recs = metrics_by_var_level_time(tgts, preds)
    if recs is None:
        print(f"  {ts}: no overlapping variables – skipped")
        continue

    for r in recs:
        r["timestamp"] = ts
    # results.extend(recs)

    print(f"  {ts}: {len(recs)} variable-level scores computed")

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(args.data_dir, f"metrics_{ts}.csv"), index=False)


    # break
# if not results:
#     raise SystemExit("No CRPS values calculated – nothing to save.")

# df = pd.DataFrame(results).sort_values("timestamp")
# df.to_csv(output_csv, index=False)

# print("\nSummary")
# print("-------")
# print(f"Saved CSV: {output_csv}")
# print(f"Mean  CRPS: {df['crps'].mean():.6f}")
# print(f"Std   CRPS: {df['crps'].std():.6f}")
# print(f"Min   CRPS: {df['crps'].min():.6f}")
# print(f"Max   CRPS: {df['crps'].max():.6f}")


# if __name__ == "__main__":
#     main()
