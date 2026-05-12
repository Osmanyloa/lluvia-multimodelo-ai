"""Core workflow for hybrid rainfall multi-model forecasting."""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import s3fs
import xarray as xr
from ecmwf.opendata import Client
from scipy.interpolate import RegularGridInterpolator


DEFAULT_MODEL_WEIGHTS = {
    "gfs": 0.35,
    "ecmwf_ifs": 0.40,
    "ecmwf_aifs": 0.15,
    "gfs_graphcast": 0.10,
}

DEFAULT_CALIBRATION_FACTORS = {
    "gfs": 1.6,
    "ecmwf_ifs": 1.6,
    "ecmwf_aifs": 2.5,
    "gfs_graphcast": 3.0,
}


@dataclass(frozen=True)
class ModelField:
    """Loaded precipitation field and its native grid."""

    dataset: xr.Dataset
    precipitation: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray

    @property
    def max_precipitation(self) -> float:
        return float(np.nanmax(self.precipitation))


class RainfallMultiModel:
    """Download, load, calibrate, and blend physical plus AI forecast models."""

    def __init__(self, base_dir: str | Path = "data/raw/models") -> None:
        self.date: str | None = None
        self.run: str | None = None
        self.accumulation: str | None = None
        self.lead_time: int | None = None
        self.step_range: str | None = None
        self.start_hour: int | None = None
        self.end_hour: int | None = None
        self.is_range = False
        self.fields: dict[str, ModelField] = {}
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def configure(
        self,
        date: str | None = None,
        run: str | None = None,
        accumulation: str | None = None,
    ) -> None:
        """Configure forecast date, cycle, and accumulation window.

        Parameters
        ----------
        date:
            Forecast initialization date in `YYYYMMDD` format.
        run:
            Forecast cycle: `00`, `06`, `12`, or `18`.
        accumulation:
            One of `12h`, `24h`, `5d`, or a custom range such as `f024-f144`.
        """
        if date is None:
            date = input("Enter the forecast date (YYYYMMDD, e.g. 20260511): ").strip()

        if run is None:
            print("\nAvailable cycles: 00, 06, 12, 18")
            run = input("Enter the forecast cycle (00/06/12/18): ").strip()

        if accumulation is None:
            print("\nAvailable accumulations:")
            print("  12h - 12-hour accumulated precipitation")
            print("  24h - 24-hour accumulated precipitation")
            print("  5d  - 5-day accumulated precipitation")
            print("  Custom range: f024-f144")
            accumulation = input("Enter accumulation (12h/24h/5d/f###-f###): ").strip().lower()

        self._validate_configuration(date, run)
        self.date = date
        self.run = run
        self.accumulation = accumulation
        self._parse_accumulation(accumulation)

        print("\nConfiguration")
        print(f"  Date: {self.date}")
        print(f"  Cycle: {self.run}Z")
        if self.is_range:
            print(f"  Accumulation: {self.accumulation} ({self.start_hour}-{self.end_hour} h)")
        else:
            print(f"  Accumulation: {self.accumulation} ({self.lead_time} h)")

    def download_gfs(self) -> None:
        """Download and load NOAA GFS accumulated precipitation."""
        self._require_configured()
        print("\nDownloading GFS...")

        model_dir = self.base_dir / "gfs"
        model_dir.mkdir(parents=True, exist_ok=True)
        base_url = (
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
            f"gfs.{self.date}/{self.run}/atmos/"
        )

        if self.is_range:
            end_path = self._download_gfs_file(model_dir, base_url, self.end_hour)
            ds_end = self._open_gfs_accumulation(end_path, f"0-{self.end_hour}")

            if self.start_hour and self.start_hour > 0:
                start_path = self._download_gfs_file(model_dir, base_url, self.start_hour)
                ds_start = self._open_gfs_accumulation(start_path, f"0-{self.start_hour}")
                precipitation = ds_end["tp"].values - ds_start["tp"].values
            else:
                precipitation = ds_end["tp"].values
            dataset = ds_end
        else:
            path = self._download_gfs_file(model_dir, base_url, self.lead_time)
            dataset = self._open_gfs_accumulation(path, self.step_range)
            precipitation = dataset["tp"].values

        self._store_model("gfs", dataset, precipitation)

    def download_ecmwf_ifs(self) -> None:
        """Download and load ECMWF IFS accumulated precipitation."""
        self._require_configured()
        print("\nDownloading ECMWF IFS...")
        model_dir = self.base_dir / "ecmwf_ifs"
        model_dir.mkdir(parents=True, exist_ok=True)
        client = Client(source="ecmwf")

        if self.is_range:
            dataset, precipitation = self._download_ecmwf_range(
                client=client,
                model_dir=model_dir,
                prefix="ifs",
                retrieve_kwargs={"stream": "oper"},
                convert_to_mm=True,
            )
        else:
            path = model_dir / f"ifs_{self.accumulation}_{self.date}_{self.run}z.grib"
            self._retrieve_if_missing(
                path,
                client,
                date=self.date,
                time=int(self.run),
                step=self.lead_time,
                param="tp",
                stream="oper",
            )
            dataset = self._open_surface_accumulation(path)
            precipitation = dataset["tp"].values * 1000.0

        self._store_model("ecmwf_ifs", dataset, precipitation)

    def download_ecmwf_aifs(self) -> None:
        """Download and load ECMWF AIFS accumulated precipitation."""
        self._require_configured()
        print("\nDownloading ECMWF AIFS...")
        model_dir = self.base_dir / "ecmwf_aifs"
        model_dir.mkdir(parents=True, exist_ok=True)
        client = Client(source="ecmwf", model="aifs-single")

        if self.is_range:
            dataset, precipitation = self._download_ecmwf_range(
                client=client,
                model_dir=model_dir,
                prefix="aifs",
                retrieve_kwargs={"type": "fc"},
                convert_to_mm=False,
            )
        else:
            path = model_dir / f"aifs_{self.accumulation}_{self.date}_{self.run}z.grib"
            self._retrieve_if_missing(
                path,
                client,
                date=self.date,
                time=int(self.run),
                step=self.lead_time,
                type="fc",
                param="tp",
            )
            dataset = self._open_surface_accumulation(path)
            precipitation = dataset["tp"].values

        self._store_model("ecmwf_aifs", dataset, precipitation)

    def download_gfs_graphcast(self) -> None:
        """Download and load GraphCast-GFS accumulated precipitation."""
        self._require_configured()
        print("\nDownloading GraphCast-GFS...")
        model_dir = self.base_dir / "gfs_graphcast"
        model_dir.mkdir(parents=True, exist_ok=True)

        forecast_hour = self.end_hour if self.is_range else self.lead_time
        path = model_dir / f"graphcast_{self.accumulation}_{self.date}_{self.run}z.grib2"

        if not path.exists():
            fs = s3fs.S3FileSystem(anon=True)
            s3_path = (
                f"noaa-nws-graphcastgfs-pds/graphcastgfs.{self.date}/{self.run}/"
                f"forecasts_13_levels/graphcastgfs.t{self.run}z.pgrb2.0p25.f{forecast_hour:03d}"
            )
            with fs.open(s3_path, "rb") as source, path.open("wb") as target:
                target.write(source.read())
            print(f"  Downloaded: {path.name}")
        else:
            print(f"  Reusing existing file: {path.name}")

        dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "surface",
                    "stepType": "accum",
                    "shortName": "tp",
                },
                "indexpath": "",
            },
        )
        self._store_model("gfs_graphcast", dataset, dataset["tp"].values)

    def load_all_models(self) -> None:
        """Download and load every configured model, skipping failed sources."""
        self._require_configured()
        print("\n" + "=" * 60)
        print("Loading all forecast models")
        print("=" * 60)

        downloaders = (
            ("GFS", self.download_gfs),
            ("ECMWF IFS", self.download_ecmwf_ifs),
            ("ECMWF AIFS", self.download_ecmwf_aifs),
            ("GraphCast-GFS", self.download_gfs_graphcast),
        )
        for label, downloader in downloaders:
            try:
                downloader()
            except Exception as exc:
                print(f"  {label} failed: {exc}")

        print("\n" + "=" * 60)
        print("Model loading finished")
        print("=" * 60)
        self.print_summary()

    def print_summary(self) -> None:
        """Print a concise summary of loaded model fields."""
        print("\nLoaded model summary")
        print(f"  Date: {self.date}")
        print(f"  Cycle: {self.run}Z")
        print(f"  Accumulation: {self.accumulation}")
        print(f"  Loaded models: {len(self.fields)}")

        for model_name, field in self.fields.items():
            print(f"\n  {model_name.upper()}")
            print(f"    Max precipitation: {field.max_precipitation:.2f} mm")
            print(f"    Shape: {field.precipitation.shape}")

    def get_precipitation(self, model_name: str) -> np.ndarray:
        """Return precipitation values for a loaded model."""
        return self._field(model_name).precipitation

    def get_coordinates(self, model_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return latitude and longitude arrays for a loaded model."""
        field = self._field(model_name)
        return field.latitude, field.longitude

    def get_dataset(self, model_name: str) -> xr.Dataset:
        """Return the native xarray dataset for a loaded model."""
        return self._field(model_name).dataset

    def create_multimodel(
        self,
        weights: dict[str, float] | None = None,
        calibration_factors: dict[str, float] | None = None,
        resolution: float = 0.25,
    ) -> xr.DataArray:
        """Interpolate loaded models to one grid and compute a weighted blend."""
        if not self.fields:
            raise ValueError("No models are loaded. Run load_all_models() first.")

        weights = dict(DEFAULT_MODEL_WEIGHTS if weights is None else weights)
        calibration_factors = dict(
            DEFAULT_CALIBRATION_FACTORS if calibration_factors is None else calibration_factors
        )
        weights = self._normalize_weights(weights)

        print(f"\nCreating common-grid multi-model ({resolution} degrees)...")
        longitude = np.arange(-180, 180, resolution)
        latitude = np.arange(-90, 90 + resolution, resolution)
        print(f"  Grid size: {len(latitude)} x {len(longitude)}")

        interpolated: dict[str, np.ndarray] = {}
        for model_name, field in self.fields.items():
            print(f"  Interpolating {model_name}...")
            values = self._interpolate_to_common_grid(model_name, longitude, latitude)
            factor = calibration_factors.get(model_name, 1.0)
            interpolated[model_name] = values * factor
            print(
                "    Native range: "
                f"{np.nanmin(field.precipitation):.2f} to {np.nanmax(field.precipitation):.2f} mm"
            )
            print(f"    Calibrated range: {np.nanmin(interpolated[model_name]):.2f} to {np.nanmax(interpolated[model_name]):.2f} mm")

        shapes = {values.shape for values in interpolated.values()}
        if len(shapes) != 1:
            raise ValueError("Interpolated models have different shapes.")

        blended = np.zeros_like(next(iter(interpolated.values())))
        for model_name, values in interpolated.items():
            weight = weights.get(model_name, 0.0)
            blended += weight * values
            print(
                f"    {model_name}: weight={weight:.2f}, "
                f"calibration={calibration_factors.get(model_name, 1.0):.2f}"
            )

        first_dataset = next(iter(self.fields.values())).dataset
        time_value = self._extract_time(first_dataset)
        multimodel = xr.DataArray(
            blended[np.newaxis, :, :],
            coords={"time": [time_value], "latitude": latitude, "longitude": longitude},
            dims=["time", "latitude", "longitude"],
            attrs={
                "long_name": f"Accumulated precipitation {self.accumulation} - multi-model blend",
                "units": "mm",
                "description": "Weighted blend on a common grid",
                "grid": f"Common {resolution} x {resolution} degree grid",
                "weights": str(weights),
                "calibration_factors": str(calibration_factors),
            },
        )

        self.multimodel = multimodel
        self.interpolated_models = interpolated
        self.common_longitude = longitude
        self.common_latitude = latitude

        print("\nMulti-model field created")
        print(f"  Min: {np.nanmin(blended):.2f} mm")
        print(f"  Max: {np.nanmax(blended):.2f} mm")
        print(f"  Mean: {np.nanmean(blended):.2f} mm")
        return multimodel

    def compare_point(self, latitude: float, longitude: float) -> float:
        """Compare every model and the blend at the nearest grid point."""
        if not hasattr(self, "multimodel"):
            raise ValueError("Create the multi-model field before comparing a point.")

        lat_index = int(np.argmin(np.abs(self.common_latitude - latitude)))
        lon_index = int(np.argmin(np.abs(self.common_longitude - longitude)))

        print(f"\nPoint comparison ({latitude:.2f}N, {longitude:.2f}E)")
        print(
            "  Nearest grid point: "
            f"({self.common_latitude[lat_index]:.2f}N, {self.common_longitude[lon_index]:.2f}E)"
        )
        print("  Model precipitation:")
        for model_name, values in self.interpolated_models.items():
            print(f"    {model_name:20s}: {values[lat_index, lon_index]:6.2f} mm")

        blended_value = float(self.multimodel.values[0, lat_index, lon_index])
        print(f"    {'MULTIMODEL':20s}: {blended_value:6.2f} mm")
        return blended_value

    def _download_gfs_file(self, model_dir: Path, base_url: str, forecast_hour: int) -> Path:
        path = model_dir / f"gfs_{self.accumulation}_{self.date}_{self.run}z_f{forecast_hour:03d}.grib2"
        if path.exists():
            print(f"  Reusing existing file: {path.name}")
            return path

        url = base_url + f"gfs.t{self.run}z.pgrb2.0p25.f{forecast_hour:03d}"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, path.open("wb") as target:
            target.write(response.read())
        print(f"  Downloaded: {path.name}")
        return path

    def _download_ecmwf_range(
        self,
        client: Client,
        model_dir: Path,
        prefix: str,
        retrieve_kwargs: dict[str, Any],
        convert_to_mm: bool,
    ) -> tuple[xr.Dataset, np.ndarray]:
        end_path = model_dir / f"{prefix}_{self.accumulation}_{self.date}_{self.run}z_f{self.end_hour:03d}.grib"
        self._retrieve_if_missing(
            end_path,
            client,
            date=self.date,
            time=int(self.run),
            step=self.end_hour,
            param="tp",
            **retrieve_kwargs,
        )
        end_dataset = self._open_surface_accumulation(end_path)

        if self.start_hour and self.start_hour > 0:
            start_path = model_dir / f"{prefix}_{self.accumulation}_{self.date}_{self.run}z_f{self.start_hour:03d}.grib"
            self._retrieve_if_missing(
                start_path,
                client,
                date=self.date,
                time=int(self.run),
                step=self.start_hour,
                param="tp",
                **retrieve_kwargs,
            )
            start_dataset = self._open_surface_accumulation(start_path)
            precipitation = end_dataset["tp"].values - start_dataset["tp"].values
        else:
            precipitation = end_dataset["tp"].values

        if convert_to_mm:
            precipitation = precipitation * 1000.0
        return end_dataset, precipitation

    @staticmethod
    def _retrieve_if_missing(path: Path, client: Client, **kwargs: Any) -> None:
        if path.exists():
            print(f"  Reusing existing file: {path.name}")
            return
        client.retrieve(target=str(path), **kwargs)
        print(f"  Downloaded: {path.name}")

    @staticmethod
    def _open_surface_accumulation(path: Path) -> xr.Dataset:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "surface", "stepType": "accum"},
                "indexpath": "",
            },
        )

    @staticmethod
    def _open_gfs_accumulation(path: Path, step_range: str | None) -> xr.Dataset:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "surface",
                    "stepType": "accum",
                    "stepRange": step_range,
                },
                "indexpath": "",
            },
        )

    def _store_model(self, model_name: str, dataset: xr.Dataset, precipitation: np.ndarray) -> None:
        field = ModelField(
            dataset=dataset,
            precipitation=precipitation,
            latitude=dataset["latitude"].values,
            longitude=dataset["longitude"].values,
        )
        self.fields[model_name] = field
        print(f"  Max precipitation: {field.max_precipitation:.2f} mm")

    def _field(self, model_name: str) -> ModelField:
        if model_name not in self.fields:
            raise ValueError(f"Model '{model_name}' is not loaded. Available: {list(self.fields)}")
        return self.fields[model_name]

    def _interpolate_to_common_grid(
        self,
        model_name: str,
        common_longitude: np.ndarray,
        common_latitude: np.ndarray,
    ) -> np.ndarray:
        field = self._field(model_name)
        precipitation = field.precipitation
        if precipitation.ndim == 3:
            precipitation = precipitation[0, :, :]

        native_longitude = field.longitude
        native_latitude = field.latitude
        if native_longitude.min() >= 0 and native_longitude.max() > 180:
            native_longitude = np.where(native_longitude > 180, native_longitude - 360, native_longitude)

        if native_longitude[0] > native_longitude[-1]:
            sort_index = np.argsort(native_longitude)
            native_longitude = native_longitude[sort_index]
            precipitation = precipitation[:, sort_index]

        if native_latitude[0] > native_latitude[-1]:
            native_latitude = native_latitude[::-1]
            precipitation = precipitation[::-1, :]

        interpolator = RegularGridInterpolator(
            (native_latitude, native_longitude),
            precipitation,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        lat_grid, lon_grid = np.meshgrid(common_latitude, common_longitude, indexing="ij")
        points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        return np.maximum(interpolator(points).reshape(len(common_latitude), len(common_longitude)), 0.0)

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        weight_sum = sum(weights.get(model_name, 0.0) for model_name in self.fields)
        if weight_sum <= 0:
            raise ValueError("Weights for loaded models must sum to a positive value.")
        if abs(weight_sum - 1.0) > 0.001:
            print(f"Weight sum is {weight_sum:.3f}; normalizing to 1.0.")
            return {model_name: weight / weight_sum for model_name, weight in weights.items()}
        return weights

    @staticmethod
    def _extract_time(dataset: xr.Dataset) -> Any:
        if hasattr(dataset, "time"):
            values = dataset.time.values
            if np.isscalar(values) or getattr(values, "ndim", 0) == 0:
                return values.item()
            return values[0]
        return np.datetime64("2024-01-01")

    @staticmethod
    def _validate_configuration(date: str, run: str) -> None:
        if len(date) != 8 or not date.isdigit():
            raise ValueError("date must use YYYYMMDD format.")
        if run not in {"00", "06", "12", "18"}:
            raise ValueError("run must be one of: 00, 06, 12, 18.")

    def _parse_accumulation(self, accumulation: str) -> None:
        self.is_range = False
        if accumulation in {"12h", "24h", "5d"}:
            lead_times = {"12h": 12, "24h": 24, "5d": 120}
            self.lead_time = lead_times[accumulation]
            self.start_hour = 0
            self.end_hour = self.lead_time
            self.step_range = f"0-{self.lead_time}"
            return

        match = re.match(r"f(\d+)-f(\d+)", accumulation)
        if not match:
            raise ValueError("accumulation must be 12h, 24h, 5d, or f###-f###.")

        self.start_hour = int(match.group(1))
        self.end_hour = int(match.group(2))
        if self.start_hour >= self.end_hour:
            raise ValueError("The ending forecast hour must be greater than the starting hour.")

        self.is_range = True
        self.lead_time = self.end_hour
        self.step_range = f"{self.start_hour}-{self.end_hour}"
