# cleaned_gencast_demo.py
# -----------------------------------------------------------------------------
# A head‑less version of the GenCast‑Mini notebook: no widgets, plots or JS.
# It
#   • loads local ERA‑5 NetCDF slices → ds  → ds_batch,
#   • loads the GenCast 1p0deg Mini <2019 model checkpoint,
#   • computes a single loss + gradient example.
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # easier to spot OOM
# # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"    # cap GPU alloc
# os.environ["JAX_LOG_LEVEL"] = "debug"                   # verbose runtime log
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

os.environ["LD_LIBRARY_PATH"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import os, dataclasses
from typing import Iterable

import haiku as hk
import jax
import numpy as np
import pandas as pd
import xarray as xr
from jax import numpy as jnp
from google.cloud import storage
import optax
# jax.config.update("jax_log_compiles", True)   # one-line compile log

import random
import wandb
from graphcast import (
    gencast, rollout, xarray_jax, normalization, checkpoint,
    data_utils, xarray_tree, denoiser, nan_cleaning
)
import functools

from jax import lax
import argparse
import pathlib
import yaml
from argparse import Namespace
import time
import dask
import glob
import re
from datetime import datetime

# import torch
# torch.multiprocessing.set_sharing_strategy("file_descriptor")


import xarray
import xbatcher
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


from gencast_train_utils import get_local_era5_for_ngcm, rearrange_for_pmap, get_batch_contiguous, get_batch
import pickle
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
os.chdir("/home/work/kaist-siml-ssd/manual_v0529/graphcast")


try:
    jax.config.update("jax_default_matmul_precision", "bfloat16")
except Exception:
    jax.config.update("jax_default_matmul_precision", "fastest")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--inference-batch_size", type=int, default=32, help="Batch size for inference")
parser.add_argument("--batch_accumulate", type=int, default=8, help="Number of gradient accumulation steps")
parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warm-up steps")
parser.add_argument("--train_steps", type=int, default=8000000, help="Total number of training steps")
parser.add_argument("--eval_freq", type=int, default=10000, help="Frequency of evaluation during training")
parser.add_argument("--peak_lr", type=float, default=1e-3, help="Peak learning rate")
parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
parser.add_argument("--test_mode", type=eval, default=False, help="test_mode")
parser.add_argument("--seed", type=int, default=None, help="seed")
parser.add_argument("--gko", type=eval, default=True)
parser.add_argument("--grad_clip", type=float, default=5.0,
                    help="Global gradient clipping threshold; 0 disables clipping")
parser.add_argument("--resume", type=eval, default=False, help="Resume from latest checkpoint in this run's folder")
parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to a specific checkpoint .pkl")
parser.add_argument("--wandb_id", type=str, default=None, help="W&B run id (use with --resume to continue the same run)")

args = parser.parse_args()
# args.test_mode = True
if args.seed is None:
    args.seed = int(time.time())
if args.test_mode:
    os.environ["WANDB_MODE"] = "disabled" 
print(args)
# -----------------------------------------------------------------------------
# 1.  Configuration  ───────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

main_key = jax.random.PRNGKey(args.seed)


def load_config(name):
    path = pathlib.Path("configs/GenCast 1p0deg <2019") / f"{name}.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

task_cfg = load_config("task_config")
sampler_cfg = load_config("sampler_config")
noise_cfg = load_config("noise_config")
noise_enc_cfg = load_config("noise_encoder_config")
denoiser_arch_cfg = load_config("denoiser_architecture_config")
# -----------------------------------------------------------------------------
# 4.  Normalisation statistics  ───────────────────────────────────────────────-
# -----------------------------------------------------------------------------

stats_dir = pathlib.Path("stats")

def load_stat(name):
    path = stats_dir / f"{name}.nc"
    ds = xr.load_dataset(path).astype(np.float32)
    return ds

diffs_stddev_by_level = load_stat("diffs_stddev_by_level")
mean_by_level = load_stat("mean_by_level")
stddev_by_level = load_stat("stddev_by_level")
min_by_level = load_stat("min_by_level")
# -----------------------------------------------------------------------------
# 5.  Local ERA‑5 helpers  ─────────────────────────────────────────────────────
# -----------------------------------------------------------------------------



# train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
#     ds_batch,
#     target_lead_times=slice("12h", "12h"),
#     **dataclasses.asdict(task_cfg),
# )

def construct_wrapped_gencast():
    predictor = gencast.GenCast(
        sampler_config=sampler_cfg,
        task_config=task_cfg,
        denoiser_architecture_config=denoiser_arch_cfg,
        noise_config=noise_cfg,
        noise_encoder_config=noise_enc_cfg,
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
        var_to_clean="sea_surface_temperature",
    )
    return predictor


@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
    predictor = construct_wrapped_gencast()
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(inputs, targets, forcings):
    predictor = construct_wrapped_gencast()
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics),
    )

def grads_fn(params, state, rng, inputs, targets, forcings):
    def _aux(p, s, k, i, t, f):
        (loss, diagnostics), nxt_state = loss_fn.apply(
            p, s, k, i, t, f
        )
        return loss, (diagnostics, nxt_state)

    (loss, (diagnostics, nxt_state)), grads = jax.value_and_grad(
        _aux, has_aux=True
    )(params, state, rng, inputs, targets, forcings)

    return loss, diagnostics, nxt_state, grads



import pickle
import pathlib



def _take_first_device(tree, n_dev: int):
    """Strip leading 'device' axis that pmap adds to outputs."""
    def _one(x):
        x = jax.device_get(x)
        if isinstance(x, np.ndarray) and x.ndim > 0 and x.shape[0] == n_dev:
            return x[0]  # first replica (all replicas identical after pmean)
        return x
    return jax.tree_util.tree_map(_one, tree)

def _norms_tree_to_named_dict(norms_tree) -> dict:
    """
    norms_tree is a Haiku-style tree: {module: {param_name: scalar}}
    → flat { "module/param_name": float }
    """
    d = hk.data_structures.to_mutable_dict(norms_tree)
    out = {}
    for mod, pd in d.items():
        for pname, v in pd.items():
            out[f"{mod}/{pname}"] = float(np.asarray(v))
    return out


def _loss_single_shard(rng, inputs, targets, forcings, params, state):
    """
    Per-device loss (no grads).  `params` and `state` are captured
    from the outer scope and therefore broadcast automatically.
    """
    (loss, diagnostics), _ = loss_fn.apply(
        params,          # ← broadcast
        state,           # ← broadcast
        rng,             # ← device-specific key
        inputs,
        targets,
        forcings,
    )

    # Average across all replicas so every host sees identical numbers
    loss        = lax.pmean(loss,        axis_name="device")
    diagnostics = lax.pmean(diagnostics, axis_name="device")
    return loss, diagnostics

def _grads_single_shard(rng, inputs, targets, forcings, params, state, opt_state):
    loss, diagnostics, nxt_state, grads = grads_fn(
        params, state, rng, inputs, targets, forcings
    )

    # Compute global gradient norm *before* clipping for logging
    grad_norm = optax.global_norm(grads)

    # Optional clipping
    if GRAD_CLIP_THRESHOLD > 0.0:
        clipping_factor = jnp.minimum(1.0, GRAD_CLIP_THRESHOLD / (grad_norm + 1e-6))
        grads = jax.tree_util.tree_map(lambda g: g * clipping_factor, grads)

    # Average across devices so that every replica sees the same numbers
    loss  = lax.pmean(loss,  axis_name="device")
    grads = lax.pmean(grads, axis_name="device")
    grad_norm  = lax.pmean(grad_norm,  axis_name="device")

    per_leaf_norms = jax.tree_util.tree_map(lambda g: jnp.linalg.norm(g), grads)

    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, diagnostics, nxt_state, grads, grad_norm, params, opt_state, per_leaf_norms

pmapped_loss = xarray_jax.pmap(
    _loss_single_shard,
    dim="device",       
    axis_name="device",
)

pmapped_grads = xarray_jax.pmap(
    _grads_single_shard,
    dim="device", 
    axis_name="device",
)
n_dev = jax.local_device_count()
def _to_cpu_numpy(tree):
    """One host copy per leaf (handles replicated or sharded arrays)."""
    def _one(x):
        x = jax.device_get(x)            # → numpy on host, keeps shape
        # replicated arrays arrive as (n_dev, ...) – strip device axis
        if isinstance(x, np.ndarray) and x.ndim>0 and x.shape[0] == n_dev:
            x = x[0]
        return x
    return jax.tree_util.tree_map(_one, tree)


def save_ckpt(path, *, params, state, opt_state, step, rng):
    ckpt = dict(
        params    = _to_cpu_numpy(params),
        state     = _to_cpu_numpy(state),
        opt_state = _to_cpu_numpy(opt_state),
        step      = int(step),
        rng       = np.asarray(rng)       # PRNGKey is just uint32[2]
    )
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)

def load_ckpt(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    # jnp.asarray converts numpy → jax.Array; replicate to all devices
    repl = lambda t: jax.device_put_replicated(
        jax.tree_util.tree_map(jnp.asarray, t), jax.local_devices()
    )
    return dict(
        params    = repl(ckpt["params"]),
        state     = repl(ckpt["state"]),
        opt_state = repl(ckpt["opt_state"]),
        step      = ckpt["step"],
        rng       = jnp.asarray(ckpt["rng"], dtype=jnp.uint32)  # back to PRNGKey
    )


if __name__ == "__main__":
    

    run_name = None
    if args.resume_ckpt:
        run_name = pathlib.Path(args.resume_ckpt).parent.name
    else:
        run_name = f"gencast_{datetime.now():%Y%m%d_%H%M%S}"


    if args.gko:
        wandb.login(key="c2314831450af6fe3943bf27fed31d547c727353", relogin=True)
    wandb.init(
        project="gencast-mini",  
        name=run_name,
        config=vars(args),
        id = args.wandb_id,
        resume="allow" if (args.wandb_id or args.resume) else None,

        # log_model=False, 
    )

    from gencast_dataset import GenCastDataset, collate_gencast
    atmospheric_features = [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    ]
    single_features = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_precipitation_12hr",
        "sea_surface_temperature",
    ]
    static_features = [
        "geopotential_at_surface",
        "land_sea_mask",
    ]
    drop_in_input_features = [
        "total_precipitation_12hr",
    ]

    drop_in_target_features = [
        'geopotential_at_surface',
        'land_sea_mask',
    ]

    BASE_DIR = '/home/work/kaist-siml-ssd/graph_weather/graph_weather/models/gencast/era5_wb2_0p25_gencast_2000_2020.zarr'
    DOWNSAMPLE = 1

    train_dataset = GenCastDataset(
            base_dir=BASE_DIR,
            # time_start=TRAIN_START,
            # time_end=TRAIN_END,
            # compute=False,
            atmospheric_features=atmospheric_features,
            single_features=single_features,
            static_features=static_features,
            drop_in_input_features=drop_in_input_features,
            drop_in_target_features=drop_in_target_features,
            min_year=2000,
            max_year=2019,
            time_step=2,
            downsample=DOWNSAMPLE,
        )
    val_dataset = GenCastDataset(
            base_dir=BASE_DIR,
            # time_start=TRAIN_START,
            # time_end=TRAIN_END,
            # compute=False,
            atmospheric_features=atmospheric_features,
            single_features=single_features,
            static_features=static_features,
            drop_in_input_features=drop_in_input_features,
            drop_in_target_features=drop_in_target_features,
            min_year=2019,
            max_year=2020,
            time_step=2,
            downsample=DOWNSAMPLE,
        )

    BATCH_SIZE = args.batch_size
    PERSISTENT_WORKERS = True
    NUM_WORKERS = 16


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=False,
        collate_fn=collate_gencast,
        multiprocessing_context='forkserver',
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=False,
        collate_fn=collate_gencast,
        multiprocessing_context='forkserver',
    )

    # init_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     collate_fn=collate_gencast
    # )
    init_data = collate_gencast([train_dataset[0]])
    init_inp, init_tgt, init_frc = init_data['inputs'], init_data['targets'], init_data['clock']

    main_key, subkey = jax.random.split(main_key)
    with jax.default_device(jax.devices("cpu")[0]):
        params, state = loss_fn.init(
            subkey, init_inp, init_tgt, init_frc
        )


    # (3·2)  Cosine‑decay LR schedule with linear warm‑up
    warmup_fn  = optax.linear_schedule(
        init_value=0.0,
        end_value=args.peak_lr,
        transition_steps=int(args.warmup_steps),
    )
    cosine_fn  = optax.cosine_decay_schedule(
        init_value=args.peak_lr,
        decay_steps=int(args.train_steps - args.warmup_steps),
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[int(args.warmup_steps)],
    )

    # (3·3)  AdamW optimiser
    optimizer = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
    )
    tx        = optax.MultiSteps(optimizer, every_k_schedule=args.batch_accumulate)
    opt_state = jax.device_put_replicated(tx.init(params), jax.local_devices())
    params = jax.device_put_replicated(params, jax.local_devices())
    state = jax.device_put_replicated(state, jax.local_devices())
    GRAD_CLIP_THRESHOLD = float(args.grad_clip)
    import gc
    gc.collect()

    # (3·4)  House‑keeping for gradient accumulation
    accum_grads: optax.Updates | None = None
    accum_counter: int = 0
    optim_step: int = 0

    checkpoint_dir = pathlib.Path("checkpoints") / wandb.run.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume or args.resume_ckpt:
        ckpt_file = pathlib.Path(args.resume_ckpt)
        if ckpt_file and ckpt_file.exists():
            print(f"[Resume] Loading checkpoint: {ckpt_file}")
            loaded = load_ckpt(ckpt_file)
            params, state, opt_state = loaded["params"], loaded["state"], loaded["opt_state"]
            optim_step = loaded["step"]          # keep LR schedule & logs consistent
            main_key   = loaded["rng"]





    start_time = time.time()
    epochs = args.train_steps // len(train_dataloader) + 1
    print(f"Training for {epochs} epochs, {len(train_dataloader)} batches per epoch.")


    if args.resume:
        start_epoch = optim_step // len(train_dataloader)
        print(f"Resuming from epoch {start_epoch}.")
    else:
        start_epoch = 0
    prev_train_losses = []

    def check_diverged(prev_train_losses, train_loss):
        if len(prev_train_losses) == 0:
            return False
        if train_loss - prev_train_losses[-1] > 0.1:
            return True
        if len(prev_train_losses) < 10:
            return False
        mean_loss = np.mean(prev_train_losses[-10:])
        std_loss = np.std(prev_train_losses[-10:])
        if train_loss > mean_loss + 2 * std_loss:
            return True
        return False

    def _t():  # high-res timer
        return time.perf_counter()

    ckpt_path = None
    after_step = _t()
    for epoch in range(start_epoch, epochs):
        if args.test_mode and (time.time() - start_time) > 60 * 60 * 3:
            exit()
        # tic = time.time()
        epoch_loss = []
        for i, batch in tqdm(enumerate(train_dataloader)):
        # for i,batch in tqdm(enumerate(train_loader)):
            # print(i)
            # batch = train_loader[i]
            start = time.perf_counter()
            load_time = start - after_step
            # print('data loading', data_iter)
            train_inputs, train_targets, train_forcings = batch['inputs'], batch['targets'], batch['clock']

            train_inputs_pm   = rearrange_for_pmap(train_inputs)
            train_targets_pm  = rearrange_for_pmap(train_targets)
            train_forcings_pm = rearrange_for_pmap(train_forcings)
            # print(train_inputs_pm)
            # print(train_targets_pm)
            # print(train_forcings_pm)

            # Inside training loop:
            key, main_key = jax.random.split(main_key)
            keys = jax.random.split(key, num=jax.local_device_count())

            # pmapped_grads = jax.pmap(
            #     _grads_single_shard,
            #     in_axes=(0, 0, 0, 0),  # inputs, targets, forcings, key
            #     axis_name="device",
            # )
            
            loss_pm, diagnostics_pm, nxt_state_pm, grads_pm, grad_norm_pm, params, opt_state, norms_pm  = pmapped_grads(
                keys, train_inputs_pm, train_targets_pm, train_forcings_pm, params, state, opt_state
            )
            state = nxt_state_pm
            norms_host = _take_first_device(norms_pm, n_dev)
            norms_named = _norms_tree_to_named_dict(norms_host)


            # step_time = time.time() - tic  
            # print(f"[iter={i+1:>7d}]  step_time={step_time:6.3f}s")
            # tic = time.time()
            lr_now = float(lr_schedule(optim_step))
            train_loss = float(jax.device_get(jax.tree_util.tree_map(lambda x: x.mean(), loss_pm)))
            grad_norm = float(jax.device_get(jax.tree_util.tree_map(lambda x: x.mean(), grad_norm_pm)))
            after_step = time.perf_counter()
            step_time = after_step - start
            # print('model_stepping', step_time)
            # print('overall', step_time + data_iter)
            # wandb.log({"train/loss": train_loss,
            #            "train/grad_norm": grad_norm,
            #            "train/step_time": step_time,
            #            "step": optim_step,
            #            "lr": lr_now,
            #            })

            log_payload = {
                "train/loss": train_loss,
                "train/grad_norm": grad_norm,
                "train/step_time": step_time,
                "train/load_time": load_time,
                "train/overall_time": load_time + step_time,
                "step": optim_step,
                "lr": lr_now,
            }
            # Prefix so charts group nicely in W&B
            for name, val in norms_named.items():
                log_payload[f"grad_norms/{name}"] = val

            wandb.log(log_payload)

            optim_step += 1
            epoch_loss.append(train_loss)
        train_loss = np.mean(epoch_loss)
        wandb.log({"train/epoch_loss": train_loss, "epoch": epoch, "step": optim_step})
        if check_diverged(prev_train_losses, train_loss) and (ckpt_path is not None):
            print(f"Loss increased by more than mean+3*std at {epoch}., rollback to the previous checkpoint {ckpt_path}.")
            # break
            prev_ckpt = load_ckpt(ckpt_path)
            params = prev_ckpt["params"]
            state = prev_ckpt["state"]
            opt_state = prev_ckpt["opt_state"]
        else:
            prev_train_losses.append(train_loss)

        val_epoch_loss = []
        if epoch in range(0, epochs, max(epochs//1000,1)):
            for i,batch in tqdm(enumerate(val_dataloader)):
                # batch = inv_scaler(batch.astype('float32'))
                val_inputs, val_targets, val_forcings = batch['inputs'], batch['targets'], batch['clock']

                val_inputs_pm   = rearrange_for_pmap(val_inputs)
                val_targets_pm  = rearrange_for_pmap(val_targets)
                val_forcings_pm = rearrange_for_pmap(val_forcings)


                key, main_key = jax.random.split(main_key)
                keys = jax.random.split(key, num=jax.local_device_count())

                loss_val, diagnostics_val = pmapped_loss(
                    keys, val_inputs_pm, val_targets_pm, val_forcings_pm, params, state
                )
                loss_val_final = jax.tree_util.tree_map(lambda x: x[0], loss_val)


            
                # --- W&B logging ---
                val_loss = float(loss_val_final)
                wandb.log({"val/loss": val_loss, "step": optim_step})
                val_epoch_loss.append(val_loss)
            wandb.log({"val/epoch_loss": np.mean(val_epoch_loss), "epoch": epoch, "step": optim_step})

            # --- Checkpoint only from host 0 (multi-host safe) ---
            if jax.process_index() == 0 and (not args.test_mode):
                ckpt_path = checkpoint_dir / f"step_{optim_step:07d}.pkl"
                save_ckpt(
                    ckpt_path,
                    params=params,
                    state=state,
                    opt_state=opt_state,
                    step=optim_step,
                    rng=main_key
                )
                    # wandb.save(str(ckpt_path))
                
                # print(f"[iter={i+1:>7d}]  step_time={step_time:6.3f}s")
                # tic = time.time()
                # print(f"Validation Loss at step {i+1}: {loss_val_final}")
