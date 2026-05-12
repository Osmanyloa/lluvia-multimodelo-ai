# Hybrid Rainfall Multi-Model AI

Portfolio-ready Python project for accumulated rainfall forecasting with a weighted blend of physical numerical weather prediction models and AI-based forecast systems.

The workflow combines sources such as GFS, ECMWF IFS, GraphCast-GFS, and ECMWF AIFS on a common grid to produce accumulated precipitation fields and regional maps for meteorological analysis.

## Features

- Downloads and reads GRIB output from multiple forecast systems.
- Supports fixed accumulations (`12h`, `24h`, `5d`) and custom ranges such as `f024-f144`.
- Interpolates every model to a shared grid.
- Builds a calibrated weighted blend of physical and AI forecast models.
- Includes regional precipitation map renderers for the Caribbean, Cuba, Central America, Hispaniola, the Lesser Antilles, Puerto Rico, Colombia/Venezuela, the United States, Florida, Texas, Mexico, Iberia, and the Canary Islands.

## Project Structure

```text
lluvia-multimodelo-ai/
├── src/rainfall_multimodel/
│   ├── core.py          # Main RainfallMultiModel workflow
│   ├── maps.py          # Regional map rendering functions
│   └── geography.py     # Geographic layer helpers
├── examples/
│   └── run_forecast.py  # Minimal usage example
├── data/                # Local weather data ignored by git
├── outputs/             # Generated maps and figures ignored by git
├── docs/
│   └── project_overview.md
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> Note: `cfgrib` requires the system `eccodes` library. On Linux, install it with `apt-get install libeccodes-dev`; on macOS, install it with `brew install eccodes`.

## Quick Start

```python
from rainfall_multimodel import RainfallMultiModel

model = RainfallMultiModel()
model.configure(date="20260511", run="06", accumulation="f024-f144")
model.load_all_models()

multi_model = model.create_multimodel(
    weights={
        "gfs": 0.25,
        "ecmwf_ifs": 0.25,
        "ecmwf_aifs": 0.25,
        "gfs_graphcast": 0.25,
    }
)

model.compare_point(latitude=40.0, longitude=-100.0)
```

## Modeling Idea

The multi-model blend combines traditional physics-based predictors with AI-based forecast systems. The initial philosophy is to balance both groups, then tune weights and calibration factors by region, data availability, and historical performance.

## Author

Osmany Lorenzo Amaro
