
import itertools
from typing import Any, Hashable, Sequence

import jax
import jax.numpy as jnp
import xarray as xr
from graphcast import xarray_jax as xj

Array = jax.Array | jnp.ndarray
Devices = Sequence[jax.Device]



import os, dataclasses
from typing import Iterable

import haiku as hk
import jax
import numpy as np
import pandas as pd
import xarray as xr
from jax import numpy as jnp
from google.cloud import storage
# jax.config.update("jax_log_compiles", True)   # one-line compile log

from graphcast import (
    gencast, rollout, xarray_jax, normalization, checkpoint,
    data_utils, xarray_tree, denoiser, nan_cleaning
)
import functools
from jax import lax
import argparse
import pathlib
import yaml


import itertools
from typing import Any, Hashable, Sequence

import jax
import jax.numpy as jnp
import xarray as xr
from graphcast import xarray_jax as xj

Array = jax.Array | jnp.ndarray
Devices = Sequence[jax.Device]

LEVEL15_IDX = [8,10,12,14,16,17,19,21,23,25,30,33,36]

def get_local_era5_for_ngcm(
    time_range: pd.DatetimeIndex,
    base_dir: str,
    compute: bool = True,
    level15: bool = False,
) -> xr.Dataset:
    """Merge hourly ERA‑5 NetCDF files under *base_dir* covering *time_range*."""
    file_list = [
        os.path.join(base_dir, t.strftime("%Y%m"), f"{t.strftime('%Y%m%d%H')}.nc")
        for t in time_range
    ]
    existing_files = [f for f in file_list if os.path.exists(f)]
    if not existing_files:
        raise FileNotFoundError(
            f"No ERA‑5 files found in {base_dir} for the specified range."
        )
    ds = xr.open_mfdataset(
        existing_files,
        combine="nested",
        concat_dim="time",
        parallel=False,
        chunks={"time":"auto"},
    )
    if level15:
        ds = ds.isel(level=LEVEL15_IDX)
    ds = ds.assign_coords(datetime=("datetime", time_range))
    ds = ds.assign_coords(time=("time", ds["datetime"].values))
    return ds.compute() if compute else ds


def get_batch(
    ds: xr.Dataset,
    centers: Iterable[pd.Timestamp],
    step_hours: tuple[int, ...] = (0, 12, 24),
    static_vars: tuple[str, ...] = ("geopotential_at_surface", "land_sea_mask"),
    downsample: int = 1,
) -> xr.Dataset:
    """Create GraphCast‑style (batch, time, level, lat, lon) sample."""
    if "time" not in ds.dims:
        raise ValueError("Input dataset must include a 'time' dimension")

    step_hours = np.asarray(step_hours, dtype=int)
    if (step_hours % 6 != 0).any():
        raise ValueError("step_hours must be multiples of 6 h (ERA‑5 native)")

    t_index = ds.indexes["time"]
    centre_ix = xr.DataArray([t_index.get_loc(pd.Timestamp(t)) for t in centers], dims="batch")
    step_ix = xr.DataArray(step_hours // 6, dims="time")
    sel = centre_ix + step_ix
    out = ds.isel(time=sel)

    if out["latitude"][0] > out["latitude"][-1]:
        out = out.isel(latitude=slice(None, None, -1))

    out = out.isel(
        latitude=slice(None, None, downsample),
        longitude=slice(None, None, downsample),
    )

    out = out.rename({"longitude": "lon", "latitude": "lat"})
    out = out.assign_coords(time=("time", pd.to_timedelta(step_hours, unit="h")))

    centre_vals = np.asarray([pd.Timestamp(t).to_datetime64() for t in centers])
    abs_dt = centre_vals[:, None] + out["time"].values[None, :]
    out = out.assign_coords(datetime=(("batch", "time"), abs_dt))
    if "datetime" in out.dims:
        out = out.drop_dims("datetime")

    for name in static_vars:
        if name in out.data_vars:
            out[name] = out[name].isel(batch=0, time=0, drop=True)

    pref = ["batch", "time", "level", "lat", "lon"]
    out = out.transpose(*[d for d in pref if d in out.dims], ...)
    return out.chunk({"batch": -1, "time": -1})


def get_batch_contiguous(
    ds: xr.Dataset,
    *,
    start_ix: int,                 # first centre as an integer on ds.time
    batch_size: int = 16,
    window: Iterable[int] = (0, 2, 4),   # arithmetic progression, units = native slots
    static_vars: tuple[str, ...] = ("geopotential_at_surface", "land_sea_mask"),
    downsample: int = 1,
) -> xr.Dataset:
    """
    Zero-copy mini-batch loader for ERA-5-style data that relies entirely on
    *strided views*.  Works when the centres are contiguous and the window is
    an arithmetic progression.

    Returns
    -------
    xr.Dataset
        Dimensions  (batch, time, level, lat, lon)  with a *view* of the data.
    """
    # ------------------------------------------------------------------ checks
    if "time" not in ds.dims:
        raise ValueError("Dataset must have a 'time' dimension")

    window = np.asarray(window, dtype=int)
    if window.ndim != 1 or len(window) == 0:
        raise ValueError("window must be a 1-D sequence with at least one element")

    # window must be an arithmetic progression
    if len(window) > 1 and not np.all(np.diff(window) == window[1] - window[0]):
        raise ValueError("window must be equally spaced (arithmetic progression)")

    step = window[1] - window[0] if len(window) > 1 else 1          # stride d
    min_off, max_off = window[0], window[-1]
    win_len = max_off - min_off + 1                                  # size of rolling window

    # ---------------------------------------------------------------- 1. slice
    block_start = start_ix + min_off
    block_stop  = start_ix + (batch_size - 1) + max_off + 1          # slice stop is exclusive
    if block_start < 0 or block_stop > ds.sizes["time"]:
        raise IndexError("Requested indices fall outside ds.time")

    slab = ds.isel(time=slice(block_start, block_stop))              # pure view

    # ------------------------------------------------------- 2. rolling view
    rolled = (
        slab
        .rolling(time=win_len, center=False, min_periods=win_len)
        .construct("t_win")                                          # view via as_strided
        .isel(time=slice(win_len - 1, win_len - 1 + batch_size))     # keep B complete windows
    )
    # dims are now ('time', 't_win', …)
    # ------------------------------------------------------- 3. pick offsets
    # Offsets we want sit at  positions   0, step, 2*step, …
    if step > 1:
        rolled = rolled.isel(t_win=slice(0, win_len, step))          # still a view
    # else step == 1 => t_win already matches window exactly

    # ------------------------------------------------------ 4. bookkeeping
    rolled = rolled.rename({"time": "batch", "t_win": "time"})        # reshape view → view

    if rolled["latitude"][0] > rolled["latitude"][-1]:
        rolled = rolled.isel(latitude=slice(None, None, -1))

    rolled = rolled.isel(
        latitude=slice(None, None, downsample),
        longitude=slice(None, None, downsample),
    ).rename({"longitude": "lon", "latitude": "lat"})

    # attach relative offsets as timedeltas (6-hour native slot)
    rolled = rolled.assign_coords(time=("time", pd.to_timedelta(window * 6, unit="h")))

    # 2. build absolute datetimes for every batch-step -----------------------
    centre_times = ds["time"].isel(time=slice(start_ix, start_ix + batch_size)).values
    # broadcast -> shape (batch, time)
    abs_dt = centre_times[:, None] + (window * np.timedelta64(6, "h"))

    rolled = rolled.assign_coords(datetime=(("batch", "time"), abs_dt))

    # in case xarray turned `datetime` into a dim (rare), drop it back:
    if "datetime" in rolled.dims:
        rolled = rolled.drop_dims("datetime")

    # drop the batch/time axes from static 2-D variables
    for name in static_vars:
        if name in rolled.data_vars:
            rolled[name] = rolled[name].isel(batch=0, time=0, drop=True)

    pref = ["batch", "time", "level", "lat", "lon"]
    return rolled.transpose(*[d for d in pref if d in rolled.dims], ...).drop_vars("batch")


def _ensure_leading(dim_name: Hashable, obj):
    """Transpose so that `dim_name` is first if it exists."""
    if dim_name in obj.dims and obj.dims[0] != dim_name:
        new_order = (dim_name,) + tuple(d for d in obj.dims if d != dim_name)
        return obj.transpose(*new_order)
    return obj


def _shard_batch_add_device(arr, n_dev):
    """Reshape (B, …) -> (n_dev, B//n_dev, …)."""
    b = arr.shape[0]
    if b % n_dev:
        raise ValueError(f"batch={b} not divisible by n_dev={n_dev}")
    return arr.reshape((n_dev, b // n_dev, *arr.shape[1:]))


def _replicate(arr, n_dev):
    """Add leading replica axis (n_dev, …)."""
    return jnp.broadcast_to(arr, (n_dev, *arr.shape))



def rearrange_for_pmap(tree: Any,
                       *,
                       batch_dim: str = "batch",
                       replica_dim: str = "device",
                       devices: Sequence[jax.Device] | None = None):
    """
    • If a leaf has `batch_dim`  → shard that axis across replicas.
    • Otherwise               → replicate the leaf for every device.
    The new leading axis is called `replica_dim`.
    """
    n_dev = len(jax.devices() if devices is None else devices)

    def _handle_da(da: xr.DataArray):
        da = _ensure_leading(batch_dim, da)
        data = xj.unwrap_data(da)

        if batch_dim in da.dims:                    # shard
            data = _shard_batch_add_device(data, n_dev)
            new_dims = (replica_dim,) + da.dims
        else:                                       # replicate
            data = _replicate(data, n_dev)
            new_dims = (replica_dim, *da.dims)


        coords_out = {k: v for k, v in da.coords.items()
                    if batch_dim not in getattr(v, 'dims', ()) and k != batch_dim}
        if batch_dim in da.dims:
            coords_out[replica_dim] = np.arange(n_dev)
            coords_out[batch_dim] = np.arange(da.sizes[batch_dim] // n_dev)


        return xr.DataArray(
            # xj.wrap(data), dims=new_dims, coords=da.coords, attrs=da.attrs
            xj.wrap(data), dims=new_dims, coords=coords_out, attrs=da.attrs
        )


    def _handle_ds(ds: xr.Dataset,
                   batch_dim: str = "batch",
                   replica_dim: str = "device",
                   ):
        new_vars = {k: _handle_da(v) for k, v in ds.data_vars.items()}
        # new_ds = xr.Dataset(new_vars)

        coords_out = {k: v for k, v in ds.coords.items()
                    if batch_dim not in getattr(v, 'dims', ()) and k != batch_dim}
        if batch_dim in ds.dims:
            coords_out[replica_dim] = np.arange(n_dev)
            coords_out[batch_dim] = np.arange(ds.sizes[batch_dim] // n_dev)


        # add device coord
        # new_ds = new_ds.assign_coords({replica_dim: np.arange(n_dev)})
        return xr.Dataset(new_vars, coords=coords_out, attrs=ds.attrs)

    def _dispatch(leaf):
        if isinstance(leaf, xr.DataArray):
            return _handle_da(leaf)
        if isinstance(leaf, xr.Dataset):
            return _handle_ds(leaf)
        if isinstance(leaf, (jax.Array, jnp.ndarray)):
            # For raw arrays treat first axis as batch if length matches
            return (_shard_batch_add_device(leaf, n_dev)
                    if leaf.shape and leaf.shape[0] % n_dev == 0
                    else _replicate(leaf, n_dev))
        return leaf

    return jax.tree_util.tree_map(
        _dispatch,
        tree,
        is_leaf=lambda v: isinstance(
            v, (xr.Dataset, xr.DataArray, jax.Array, jnp.ndarray)),
    )

