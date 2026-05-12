"""Geographic layer helpers for precipitation map rendering."""

from pathlib import Path

import geopandas as gpd


def load_natural_earth_layers(
    admin_boundaries_path: str | Path,
    countries_path: str | Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load Natural Earth administrative and country boundary shapefiles."""
    admin_boundaries = gpd.read_file(admin_boundaries_path)
    countries = gpd.read_file(countries_path)
    return admin_boundaries, countries
