from __future__ import annotations
from typing import Dict, Any, List

from torch.utils.data import Dataset

import einops
import numpy as np
import pandas as pd
import xarray as xr
import warnings


import xbatcher as xb



class GenCastDataset(Dataset):
    """
    Minimal-CPU dataset for GenCast.

    What it does:
      - Opens an ERA5 Zarr lazily.
      - Uses xbatcher to index windows over time: [t, t+Δ, t+2Δ].
      - __getitem__ ONLY slices & loads the needed window from Zarr and returns
        raw numpy arrays (no normalization, no residuals, no noise, no clock features,
        no concatenation/rearranges).

    What it DOES NOT do (push these to the model/wrapper on device):
      - normalization / standardization
      - time embeddings (sin/cos)
      - target residuals (y - x_latest)
      - noise corruption
      - channel packing / transposes / concatenations

    Returned sample (all numpy arrays):
      {
        "inputs_atm":   {var: (time=2, level, lat, lon)},
        "inputs_single":{var: (time=2, lat, lon)},
        "inputs_static":{var: (lat, lon) or (time=2, lat, lon) if time-dim present},
        "target_atm":   {var: (level, lat, lon)},
        "target_single":{var: (lat, lon)},
        "times": {"inputs": np.datetime64[2], "target": np.datetime64[]},
      }
    """

    def __init__(
        self,
        base_dir: str,
        atmospheric_features: List[str],
        single_features: List[str],
        static_features: List[str],
        drop_in_input_features: List[str] = [],
        drop_in_target_features: List[str] = [],
        min_year: int = 2020,
        max_year: int = 2018,
        time_step: int = 2,
        downsample: int = 1,
        consolidated: bool = False,
    ):
        super().__init__()

        # --- config ---
        self.atmospheric_features = list(atmospheric_features)
        self.single_features = list(single_features)
        self.static_features = list(static_features)
        self.drop_in_input_features = list(drop_in_input_features)
        self.drop_in_target_features = list(drop_in_target_features)

        self.max_year = int(max_year)
        self.min_year = int(min_year)
        self.time_step = int(time_step)
        self.downsample = int(downsample)

        # --- open zarr lazily ---
        ds = xr.open_zarr(base_dir, consolidated=consolidated)

        # --- spatial downsample view (no compute yet) ---
        lon_slice = slice(0, None, self.downsample)
        lat_slice = slice(0, None, self.downsample)
        ds = ds.isel(longitude=lon_slice, latitude=lat_slice)

        # keep grid handy (read once)
        self.grid_lon = np.asarray(ds["longitude"].values)
        self.grid_lat = np.asarray(ds["latitude"].values)

        # optional: pressure levels if present
        self.pressure_levels = (
            np.asarray(ds["level"].values) if "level" in ds.dims or "level" in ds.coords else None
        )

        # time mask: training years strictly < max_year
        time_mask = (self.min_year <=ds["time"].dt.year)&(ds["time"].dt.year < self.max_year)
        ds = ds.isel(time=time_mask)

        # restrict to only variables we actually read
        used_vars = set(self.atmospheric_features) | set(self.single_features) | set(self.static_features)
        # (xarray drops silently if a name is missing; you may want assert checks)
        ds = ds[list(used_vars)]

        # xbatcher window: [t, t+Δ, t+2Δ]
        win_time = 2 * self.time_step + 1
        over_time = win_time - 1  # stride 1

        # full-globe windows; if you want patches later, add longitude/latitude sizes < full
        self._bgen = xb.BatchGenerator(
            ds,
            input_dims={
                "time": win_time,
                "longitude": ds.sizes["longitude"],
                "latitude": ds.sizes["latitude"],
            },
            input_overlap={
                "time": over_time,
                "longitude": 0,
                "latitude": 0,
            },
            preload_batch=False,  # keep it lazy; Zarr read happens in __getitem__
        )
        self.output_features_dim = len(atmospheric_features) * len(self.pressure_levels) + len(
            single_features
        )
        self.input_features_dim = self.output_features_dim + len(static_features) + 4  # +4 clock


    def __len__(self) -> int:
        return len(self._bgen)
    
    @staticmethod
    def _sin_cos01(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Given values normalized to [0,1), return (sin, cos) of 2πx as float32."""
        x = x.astype(np.float32, copy=False)
        ang = 2.0 * np.pi * x
        return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)
    
    def _get_clock_features(
        self,
        t_coord: xr.DataArray,   # dims: ("time",)
        lon_coord: xr.DataArray  # dims: ("lon",)
    ) -> xr.Dataset:
        """Return a dataset with four clock features for given time(s) and longitudes."""
        # --- year progress (time,) ---
        doy = t_coord.dt.dayofyear.values.astype(np.float32)   # [1..366], we’ll treat as /365
        year_frac = doy / 365.0
        ysin, ycos = self._sin_cos01(year_frac)
        year_progress_sin = xr.DataArray(
            ysin, dims=("time",), coords={"time": t_coord}, name="year_progress_sin"
        )
        year_progress_cos = xr.DataArray(
            ycos, dims=("time",), coords={"time": t_coord}, name="year_progress_cos"
        )

        # --- day progress in local mean time (time, lon) ---
        lon_vals = np.asarray(lon_coord.values)
        hour = t_coord.dt.hour.values.astype(np.float32)       # (time,)
        hour_grid = np.repeat(hour[:, None], repeats=lon_vals.shape[0], axis=1)  # (time, lon)
        lmt = hour_grid + (lon_vals[None, :] * (4.0 / 60.0))   # 4 min/deg -> hours
        lmt = np.mod(lmt, 24.0)
        day_frac = lmt / 24.0
        dsin, dcos = self._sin_cos01(day_frac)
        day_progress_sin = xr.DataArray(
            dsin, dims=("time", "lon"), coords={"time": t_coord, "lon": lon_coord},
            name="day_progress_sin"
        )
        day_progress_cos = xr.DataArray(
            dcos, dims=("time", "lon"), coords={"time": t_coord, "lon": lon_coord},
            name="day_progress_cos"
        )

        return xr.Dataset(
            {
                "year_progress_sin": year_progress_sin,
                "year_progress_cos": year_progress_cos,
                "day_progress_sin": day_progress_sin,
                "day_progress_cos": day_progress_cos,
            }
        )
    

    @staticmethod
    def _to_rel_timedelta(t: xr.DataArray, ref: np.datetime64) -> np.ndarray:
        """
        Convert absolute datetime64 to timedelta64[ns] relative to `ref`.
        Ensures dtype is exactly 'timedelta64[ns]'.
        """
        t_ns = np.asarray(t.values).astype('datetime64[ns]')
        ref_ns = np.asarray(ref).astype('datetime64[ns]')
        return (t_ns - ref_ns).astype('timedelta64[ns]')
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1) window [t, t+Δ, t+2Δ]
        window: xr.Dataset = self._bgen[idx]

        # 2) split into inputs (t, t+Δ) and target (t+2Δ)
        ds_inputs = window.isel(time=[0, self.time_step])
        ds_target = window.isel(time=[-1])

        # Drop specified features
        for feature in self.drop_in_input_features:
            if feature in ds_inputs:
                ds_inputs = ds_inputs.drop_vars(feature)
        for feature in self.drop_in_target_features:
            if feature in ds_target:
                ds_target = ds_target.drop_vars(feature)

        # Materialize the two slices
        ds_inputs = ds_inputs.compute()
        ds_target = ds_target.compute()

        # Standardize dim names
        ds_inputs = ds_inputs.rename({"longitude": "lon", "latitude": "lat"})
        ds_target = ds_target.rename({"longitude": "lon", "latitude": "lat"})

        # Flip latitude if descending
        if "lat" in ds_inputs.dims:
            lat_vals = ds_inputs["lat"].values
            if lat_vals.size >= 2 and (lat_vals[1] - lat_vals[0]) < 0:
                ds_inputs = ds_inputs.isel(lat=slice(None, None, -1))
                ds_target = ds_target.isel(lat=slice(None, None, -1))

        # --- clock features ---
        # Add clock features to *inputs* (two times)
        input_clocks = self._get_clock_features(ds_inputs["time"], ds_inputs["lon"])
        ds_inputs = ds_inputs.assign(**{k: input_clocks[k] for k in input_clocks.data_vars})

        # Prepare *output* (target) clock features for the third return value
        ds_clock = self._get_clock_features(ds_target["time"], ds_target["lon"])
        ref_time = np.asarray(ds_inputs["time"].values[-1]).astype('datetime64[ns]')  # t+Δ

        ds_inputs = ds_inputs.assign_coords(
            time=self._to_rel_timedelta(ds_inputs["time"], ref_time)
        )
        ds_target = ds_target.assign_coords(
            time=self._to_rel_timedelta(ds_target["time"], ref_time)
        )
        ds_clock = ds_clock.assign_coords(
            time=self._to_rel_timedelta(ds_clock["time"], ref_time)
        )
        
        return {
            "inputs": ds_inputs,
            "targets": ds_target,
            "clock": ds_clock,   # now uses OUTPUT time (t+2Δ), not input times
        }


    # def __getitem__(self, idx: int) -> Dict[str, Any]:
    #     # 1) grab window [t, t+Δ, t+2Δ]
    #     window: xr.Dataset = self._bgen[idx]  # dims: time=2Δ+1, latitude, longitude, [level?]

    #     # 2) split into inputs (two steps) and target (last step)
    #     ds_inputs = window.isel(time=[0, self.time_step])  # t, t+Δ
    #     # Drop specified input features if present
    #     for feature in self.drop_in_input_features:
    #         if feature in ds_inputs:
    #             ds_inputs = ds_inputs.drop_vars(feature)
    #     ds_inputs = ds_inputs.compute()
    #     ds_target = window.isel(time=[-1]).compute()                   # t+2Δ
    #     time = ds_inputs["time"]
    #     ds_inputs = ds_inputs.rename({"longitude": "lon", "latitude": "lat"})
    #     ds_target = ds_target.rename({"longitude": "lon", "latitude": "lat"})

    #     return {
    #         "inputs": ds_inputs,
    #         "targets": ds_target,
    #         "time": time,
    #     }
# Put this at module top-level (so it’s picklable by DataLoader workers)
from typing import List, Dict, Any
import numpy as np
import xarray as xr

def collate_gencast(batch: List[Dict[str, Any]]) -> Dict[str, xr.Dataset]:
    """
    Collate function for GenCastDataset.
    - Accepts a list of samples like:
        {"inputs": xr.Dataset, "targets": xr.Dataset, "clock": xr.Dataset}
    - Returns batched xr.Datasets with a new leading 'batch' dimension.
    - Standardizes `time` coords to integer indices to avoid alignment issues.
    - Squeezes `time` from `targets` (since it's a single step).
    """
    inputs_list = []
    targets_list = []
    clock_list = []

    for sample in batch:
        ds_in: xr.Dataset = sample["inputs"]
        ds_tg: xr.Dataset = sample["targets"]
        ds_ck: xr.Dataset = sample["clock"]

        inputs_list.append(ds_in)
        targets_list.append(ds_tg)
        clock_list.append(ds_ck)

    # Create an explicit batch coordinate
    batch_coord = xr.DataArray(
        np.arange(len(batch), dtype=np.int32), dims=("batch",), name="batch"
    )

    inputs_b = xr.concat(inputs_list, dim=batch_coord)
    targets_b = xr.concat(targets_list, dim=batch_coord)
    clock_b = xr.concat(clock_list, dim=batch_coord)

    return {"inputs": inputs_b, "targets": targets_b, "clock": clock_b}



# from __future__ import annotations
# from typing import Dict, Any, List

# from torch.utils.data import Dataset

# import einops
# import numpy as np
# import pandas as pd
# import xarray as xr
# import warnings


# import xbatcher as xb



# class GenCastDataset(Dataset):
#     """
#     Minimal-CPU dataset for GenCast.

#     What it does:
#       - Opens an ERA5 Zarr lazily.
#       - Uses xbatcher to index windows over time: [t, t+Δ, t+2Δ].
#       - __getitem__ ONLY slices & loads the needed window from Zarr and returns
#         raw numpy arrays (no normalization, no residuals, no noise, no clock features,
#         no concatenation/rearranges).

#     What it DOES NOT do (push these to the model/wrapper on device):
#       - normalization / standardization
#       - time embeddings (sin/cos)
#       - target residuals (y - x_latest)
#       - noise corruption
#       - channel packing / transposes / concatenations

#     Returned sample (all numpy arrays):
#       {
#         "inputs_atm":   {var: (time=2, level, lat, lon)},
#         "inputs_single":{var: (time=2, lat, lon)},
#         "inputs_static":{var: (lat, lon) or (time=2, lat, lon) if time-dim present},
#         "target_atm":   {var: (level, lat, lon)},
#         "target_single":{var: (lat, lon)},
#         "times": {"inputs": np.datetime64[2], "target": np.datetime64[]},
#       }
#     """

#     def __init__(
#         self,
#         base_dir: str,
#         atmospheric_features: List[str],
#         single_features: List[str],
#         static_features: List[str],
#         drop_in_input_features: List[str] = [],
#         drop_in_target_features: List[str] = [],
#         min_year: int = 2020,
#         max_year: int = 2018,
#         time_step: int = 2,
#         downsample: int = 1,
#         consolidated: bool = False,
#     ):
#         super().__init__()

#         # --- config ---
#         self.atmospheric_features = list(atmospheric_features)
#         self.single_features = list(single_features)
#         self.static_features = list(static_features)
#         self.drop_in_input_features = list(drop_in_input_features)
#         self.drop_in_target_features = list(drop_in_target_features)

#         self.max_year = int(max_year)
#         self.min_year = int(min_year)
#         self.time_step = int(time_step)
#         self.downsample = int(downsample)

#         # --- open zarr lazily ---
#         ds = xr.open_zarr(base_dir, consolidated=consolidated)

#         # --- spatial downsample view (no compute yet) ---
#         lon_slice = slice(0, None, self.downsample)
#         lat_slice = slice(0, None, self.downsample)
#         ds = ds.isel(longitude=lon_slice, latitude=lat_slice)

#         # keep grid handy (read once)
#         self.grid_lon = np.asarray(ds["longitude"].values)
#         self.grid_lat = np.asarray(ds["latitude"].values)

#         # optional: pressure levels if present
#         self.pressure_levels = (
#             np.asarray(ds["level"].values) if "level" in ds.dims or "level" in ds.coords else None
#         )

#         # time mask: training years strictly < max_year
#         time_mask = (self.min_year <=ds["time"].dt.year)&(ds["time"].dt.year < self.max_year)
#         ds = ds.isel(time=time_mask)

#         # restrict to only variables we actually read
#         used_vars = set(self.atmospheric_features) | set(self.single_features) | set(self.static_features)
#         # (xarray drops silently if a name is missing; you may want assert checks)
#         ds = ds[list(used_vars)]

#         # xbatcher window: [t, t+Δ, t+2Δ]
#         win_time = 2 * self.time_step + 1
#         over_time = win_time - 1  # stride 1

#         # full-globe windows; if you want patches later, add longitude/latitude sizes < full
#         self._bgen = xb.BatchGenerator(
#             ds,
#             input_dims={
#                 "time": win_time,
#                 "longitude": ds.sizes["longitude"],
#                 "latitude": ds.sizes["latitude"],
#             },
#             input_overlap={
#                 "time": over_time,
#                 "longitude": 0,
#                 "latitude": 0,
#             },
#             preload_batch=False,  # keep it lazy; Zarr read happens in __getitem__
#         )
#         self.output_features_dim = len(atmospheric_features) * len(self.pressure_levels) + len(
#             single_features
#         )
#         self.input_features_dim = self.output_features_dim + len(static_features) + 4  # +4 clock


#     def __len__(self) -> int:
#         return len(self._bgen)
    
#     @staticmethod
#     def _sin_cos01(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         """Given values normalized to [0,1), return (sin, cos) of 2πx as float32."""
#         x = x.astype(np.float32, copy=False)
#         ang = 2.0 * np.pi * x
#         return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         # 1) grab window [t, t+Δ, t+2Δ]
#         window: xr.Dataset = self._bgen[idx]  # dims: time=2Δ+1, latitude, longitude, [level?]

#         # 2) split into inputs (two steps) and target (last step)
#         ds_inputs = window.isel(time=[0, self.time_step])  # t, t+Δ
#         ds_target = window.isel(time=[-1])                  # t+2Δ

#         # Drop specified input & target features if present
#         for feature in self.drop_in_input_features:
#             if feature in ds_inputs:
#                 ds_inputs = ds_inputs.drop_vars(feature)
#         for feature in self.drop_in_target_features:
#             if feature in ds_target:
#                 ds_target = ds_target.drop_vars(feature)

#         # Materialize the two slices we need
#         ds_inputs = ds_inputs.compute()
#         ds_target = ds_target.compute()  # t+2Δ

#         # Standardize dim names
#         ds_inputs = ds_inputs.rename({"longitude": "lon", "latitude": "lat"})
#         ds_target = ds_target.rename({"longitude": "lon", "latitude": "lat"})

#         # Flip latitude only if it's descending (lat[1] - lat[0] < 0)
#         if "lat" in ds_inputs.dims:
#             lat_vals = ds_inputs["lat"].values
#             if lat_vals.size >= 2 and (lat_vals[1] - lat_vals[0]) < 0:
#                 # Reverse along 'lat' for both inputs and target
#                 ds_inputs = ds_inputs.isel(lat=slice(None, None, -1))
#                 ds_target = ds_target.isel(lat=slice(None, None, -1))

#         # ---- NEW: add clock features to ds_inputs ----
#         # coords we need
#         t_coord = ds_inputs["time"]                      # (time,)
#         lon_vals = ds_inputs["lon"].values              # (lon,)
#         # 1) Year progress (global): (time,)
#         #    Use day-of-year / 365.0 like your reference code
#         doy = t_coord.dt.dayofyear.values.astype(np.float32)           # (time,)
#         year_frac = doy / 365.0                                        # (time,)
#         ysin, ycos = self._sin_cos01(year_frac)                        # each (time,)

#         year_progress_sin = xr.DataArray(
#             ysin, dims=("time",), coords={"time": t_coord}, name="year_progress_sin"
#         )
#         year_progress_cos = xr.DataArray(
#             ycos, dims=("time",), coords={"time": t_coord}, name="year_progress_cos"
#         )

#         # 2) Day progress in *local mean time*: (time, lon)
#         hour = t_coord.dt.hour.values.astype(np.float32)               # (time,)
#         hour_grid = np.repeat(hour[:, None], repeats=lon_vals.shape[0], axis=1)  # (time, lon)
#         lmt = hour_grid + (lon_vals[None, :] * (4.0 / 60.0))           # add 4 min/deg -> hours
#         lmt = np.mod(lmt, 24.0)                                        # wrap to [0,24)
#         day_frac = lmt / 24.0                                          # (time, lon)
#         dsin, dcos = self._sin_cos01(day_frac)                         # each (time, lon)

#         day_progress_sin = xr.DataArray(
#             dsin, dims=("time", "lon"), coords={"time": t_coord, "lon": ds_inputs["lon"]},
#             name="day_progress_sin"
#         )
#         day_progress_cos = xr.DataArray(
#             dcos, dims=("time", "lon"), coords={"time": t_coord, "lon": ds_inputs["lon"]},
#             name="day_progress_cos"
#         )

#         # Attach into ds_inputs
#         ds_inputs = ds_inputs.assign(
#             year_progress_sin=year_progress_sin,
#             year_progress_cos=year_progress_cos,
#             day_progress_sin=day_progress_sin,
#             day_progress_cos=day_progress_cos,
#         )
#         ds_clock = xr.Dataset(
#             {
#                 "year_progress_sin": year_progress_sin,
#                 "year_progress_cos": year_progress_cos,
#                 "day_progress_sin": day_progress_sin,
#                 "day_progress_cos": day_progress_cos,
#             }
#         )

#         return {
#             "inputs": ds_inputs,
#             "targets": ds_target,
#             "clock": ds_clock,
#         }

#     # def __getitem__(self, idx: int) -> Dict[str, Any]:
#     #     # 1) grab window [t, t+Δ, t+2Δ]
#     #     window: xr.Dataset = self._bgen[idx]  # dims: time=2Δ+1, latitude, longitude, [level?]

#     #     # 2) split into inputs (two steps) and target (last step)
#     #     ds_inputs = window.isel(time=[0, self.time_step])  # t, t+Δ
#     #     # Drop specified input features if present
#     #     for feature in self.drop_in_input_features:
#     #         if feature in ds_inputs:
#     #             ds_inputs = ds_inputs.drop_vars(feature)
#     #     ds_inputs = ds_inputs.compute()
#     #     ds_target = window.isel(time=[-1]).compute()                   # t+2Δ
#     #     time = ds_inputs["time"]
#     #     ds_inputs = ds_inputs.rename({"longitude": "lon", "latitude": "lat"})
#     #     ds_target = ds_target.rename({"longitude": "lon", "latitude": "lat"})

#     #     return {
#     #         "inputs": ds_inputs,
#     #         "targets": ds_target,
#     #         "time": time,
#     #     }


# from typing import List, Dict, Any
# import numpy as np
# import xarray as xr

# def collate_gencast(batch: List[Dict[str, Any]]) -> Dict[str, xr.Dataset]:
#     """
#     Collate function for GenCastDataset.
#     - Accepts a list of samples like:
#         {"inputs": xr.Dataset, "targets": xr.Dataset, "clock": xr.Dataset}
#     - Returns batched xr.Datasets with a new leading 'batch' dimension.
#     - Standardizes `time` coords to integer indices to avoid alignment issues.
#     - Squeezes `time` from `targets` (since it's a single step).
#     """
#     inputs_list = []
#     targets_list = []
#     clock_list = []

#     for sample in batch:
#         ds_in: xr.Dataset = sample["inputs"]
#         ds_tg: xr.Dataset = sample["targets"]
#         ds_ck: xr.Dataset = sample["clock"]

#         # Make time coordinates consistent across the batch (0..T-1)
#         if "time" in ds_in.dims:
#             ds_in = ds_in.assign_coords(time=np.arange(ds_in.sizes["time"], dtype=np.int32))
#         if "time" in ds_tg.dims:
#             ds_tg = ds_tg.assign_coords(time=np.arange(ds_tg.sizes["time"], dtype=np.int32))
#         if "time" in ds_ck.dims:
#             ds_ck = ds_ck.assign_coords(time=np.arange(ds_ck.sizes["time"], dtype=np.int32))

#         inputs_list.append(ds_in)
#         targets_list.append(ds_tg)
#         clock_list.append(ds_ck)

#     # Create an explicit batch coordinate
#     batch_coord = xr.DataArray(
#         np.arange(len(batch), dtype=np.int32), dims=("batch",), name="batch"
#     )

#     inputs_b = xr.concat(inputs_list, dim=batch_coord)
#     targets_b = xr.concat(targets_list, dim=batch_coord)
#     clock_b = xr.concat(clock_list, dim=batch_coord)

#     return {"inputs": inputs_b, "targets": targets_b, "clock": clock_b}
