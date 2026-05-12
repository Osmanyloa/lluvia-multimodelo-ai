"""Utilidades para cargar capas geograficas usadas por los mapas."""

from pathlib import Path

import geopandas as gpd


def cargar_capas_natural_earth(estados_path: str | Path, paises_path: str | Path):
    """Carga divisiones administrativas y fronteras de paises.

    Parameters
    ----------
    estados_path:
        Ruta al shapefile de Natural Earth `ne_10m_admin_1_states_provinces.shp`.
    paises_path:
        Ruta al shapefile de Natural Earth `ne_110m_admin_0_countries.shp`.
    """
    estados = gpd.read_file(estados_path)
    paises = gpd.read_file(paises_path)
    return estados, paises
