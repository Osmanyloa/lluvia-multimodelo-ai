# Technical Overview

## Problem

Forecasting accumulated rainfall benefits from combining models with different strengths. Physics-based models represent atmospheric dynamics through numerical equations, while AI-based systems can capture learned patterns from large historical datasets.

## Workflow

1. Configure forecast date, cycle, and accumulation window.
2. Download or locate GRIB files for each model.
3. Load accumulated precipitation from every source.
4. Interpolate all fields to a common grid.
5. Apply model-level calibration factors.
6. Compute the weighted multi-model blend.
7. Generate regional maps and point comparisons.

## Included Models

- GFS: NOAA global physics-based forecast model.
- ECMWF IFS: ECMWF global physics-based forecast model.
- GraphCast-GFS: AI-based forecast output available from public NOAA buckets.
- ECMWF AIFS: ECMWF AI-based forecast system.

## Expected Outputs

- `xarray.DataArray` with blended accumulated precipitation.
- Regional accumulated rainfall maps.
- Point-level model comparisons for quick validation.
