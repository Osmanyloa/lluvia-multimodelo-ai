"""Funciones de mapas de precipitacion acumulada por region.

Cada funcion espera un `xarray.DataArray` con coordenadas `longitude` y `latitude`.
Algunas regiones requieren un GeoDataFrame de divisiones administrativas.
"""

from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("outputs/maps")
DEFAULT_LOGO_PATH = Path("assets/logo.png")
DEFAULT_SPAIN_LOGO_PATH = Path("assets/logo_spain.png")
DEFAULT_US_COUNTY_SHAPEFILE = Path("data/raw/shapefiles/cb_2018_us_county_500k.shp")

#-----CUBA - PRECIPITACIÓN MULTIMODELO - REJILLA COMÚN ----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
import xarray as xr
from matplotlib.patches import FancyBboxPatch
from IPython.display import display

def graficar_precipitacion_cuba(precip_dataarray, cuba=None, gdf_estados=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para Cuba y guarda la figura en outputs/maps.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    """
    # --- PARÁMETROS DE ÁREA ---
    lon_min_deg, lon_max_deg = -87, -73
    lat_min, lat_max = 18, 25


    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f"  LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f"  LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f"  Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max, lat_min),
                                           longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max),
                                           longitude=slice(lon_min_deg, lon_max_deg))

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f"  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f"  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f"  Precipitación: Min={precip_vals.min():.2f}, Max={precip_vals.max():.2f}, Mean={precip_vals.mean():.2f} mm")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 500)
    lat_fine = np.linspace(lat.min(), lat.max(), 500)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA ---
    umbral_precip = 3
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_fine_smooth)
    print(precip_max)
    if precip_max < 50:
        vmin_colorbar, vmax_colorbar = 1, 50
    elif 50 <= precip_max < 100:
        vmin_colorbar, vmax_colorbar = 0, 100
    else:
        vmin_colorbar, vmax_colorbar = 0, 150

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    if cuba is None and gdf_estados is not None:
        cuba = gdf_estados[gdf_estados['admin'] == 'Cuba'].copy()
    elif cuba is None:
        raise ValueError("Debe proporcionar 'cuba' o 'gdf_estados'")
    cuba.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=5)

    # --- COLORES ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTA CON SOMBRA ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- BARRA DE COLOR ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.015, 0.088, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(mm)", fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES ---
    ciudades_principales = {
    "PR": {'lat': 22.42, 'lon': -83.70},
    "HAB": {'lat': 23.17, 'lon': -82.28},
    "MTZ": {'lat': 23.04, 'lon': -81.58},
    "Nueva Gerona": {'lat': 21.88, 'lon': -82.80},
    "Colón": {'lat': 22.72, 'lon': -80.91},
    "SC": {'lat': 22.41, 'lon': -79.97},
    "SSP": {'lat': 21.93, 'lon': -79.43},
    "CAV": {'lat': 21.84, 'lon': -78.76},
    "CMG": {'lat': 21.38, 'lon': -77.91},
    "Moa": {'lat': 20.66, 'lon': -74.945},
    "HOL": {'lat': 20.89, 'lon': -76.26},
    "Bayamo": {'lat': 20.38, 'lon': -76.64},
    "SCU": {'lat': 20.02, 'lon': -75.82},
    "Maisí": {'lat': 20.24, 'lon': -74.15},
    "LTU": {'lat': 20.96, 'lon': -76.95},
    "CFG": {'lat': 22.15, 'lon': -80.44},
    "ART": {'lat': 22.81, 'lon': -82.76},

}

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        ax.text(
            lon_ci, lat_ci - 0.08, nombre,
            fontsize=9, color='#2C3E50', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                PathEffects.Normal()
            ]
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.05, 0.03, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print("Logo no encontrado")

    # Crédito
    ax.text(0.5, 0.39, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/cuba_mapa_precipitacion_multimodelo.png", dpi=800, bbox_inches='tight')

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax

# --- USO ---
cuba = gdf_estados[gdf_estados['admin'] == 'Cuba'].copy()
fig, ax = graficar_precipitacion_dataarray(ds_multi, cuba=cuba)


#-----CENTROAMÉRICA - PRECIPITACIÓN MULTIMODELO - REJILLA COMÚN ----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
import xarray as xr
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_centroamerica(precip_dataarray, gdf_estados=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para Centroamérica y muestra automáticamente la figura.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con todos los estados
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """

    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # Filtrar países de Centroamérica
    paises = ['Guatemala', 'Belize', 'Honduras', 'El Salvador', 'Nicaragua', 'Costa Rica', 'Panama']
    gdf_ca = gdf_estados[gdf_estados['admin'].isin(paises)].copy()

    # --- PARÁMETROS DE ÁREA ---
    lon_min_deg, lon_max_deg = -94, -74
    lat_min, lat_max = 7, 18.7

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f" LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f" LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f" Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_max+1, lat_min),
            longitude=slice(lon_min_deg, lon_max_deg)
        )
    else:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_min, lat_max+1),
            longitude=slice(lon_min_deg, lon_max_deg)
        )

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f" LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f" LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f" Precipitación: Min={precip_vals.min():.2f}, Max={precip_vals.max():.2f}, Mean={precip_vals.mean():.2f} mm")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 500)
    lat_fine = np.linspace(lat.min(), lat.max(), 500)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)

    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA SOLO POR UMBRAL (sin restringir a tierra) ---
    umbral_precip = 3
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_masked_fine)

    if precip_max < 50:
        vmin_colorbar, vmax_colorbar = 1, 50
    elif 50 <= precip_max < 100:
        vmin_colorbar, vmax_colorbar = 0, 100
    elif 100 <= precip_max < 150:
        vmin_colorbar, vmax_colorbar = 0, 150
    else:
        vmin_colorbar, vmax_colorbar = 1, 250

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)

    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )

    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTAS CON SOMBRA 3D ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- FRONTERAS PAÍSES CON EFECTO 3D ---
    for pais in paises:
        gdf_pais = gdf_ca[gdf_ca['admin'] == pais]
        if len(gdf_pais) > 0:
            fronteras = gdf_pais.unary_union.boundary
            ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                              edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                              path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                              zorder=6)
            ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                              edgecolor='white', facecolor='none', linewidth=1.3,
                              path_effects=[PathEffects.Normal()], zorder=7)

    # --- PROVINCIAS INTERNAS ---
    for pais in paises:
        provincias = gdf_ca[(gdf_ca['admin'] == pais) & (gdf_ca['type'] != 'Country')]
        if len(provincias) > 0:
            provincias.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35,
                                      zorder=4, transform=ccrs.PlateCarree())

    # --- BARRA DE COLOR ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.07, 0.088, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(mm)", fontsize=11, weight='bold')

    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])

    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)

    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES PRINCIPALES DE CENTROAMÉRICA ---
    ciudades_principales = {
        # Guatemala
        "Guatemala": {'lat': 14.63, 'lon': -90.53},
        "Quetzaltenango": {'lat': 14.83, 'lon': -91.52},
        "Cobán": {'lat': 15.47, 'lon': -90.37},
        "Flores": {'lat': 16.91, 'lon': -89.89},

        # Honduras
        "Tegucigalpa": {'lat': 14.10, 'lon': -87.22},
        "San Pedro Sula": {'lat': 15.50, 'lon': -88.03},
        "La Ceiba": {'lat': 15.75, 'lon': -86.79},
        "Choluteca": {'lat': 13.30, 'lon': -87.20},
        "Tocoa": {'lat': 15.65, 'lon': -85.99},

        # El Salvador
        "San Salvador": {'lat': 13.69, 'lon': -89.20},
        "San Miguel": {'lat': 13.48, 'lon': -88.18},

        # Nicaragua
        "Managua": {'lat': 12.13, 'lon': -86.25},
        "León": {'lat': 12.44, 'lon': -86.88},
        "Estelí": {'lat': 13.09, 'lon': -86.36},
        "Rivas": {'lat': 11.43, 'lon': -85.82},
        "Bilwi": {'lat': 14.03, 'lon': -83.39},
        "San Andrés": {'lat': 12.5830, 'lon': -81.7},
        "Providencia": {'lat': 13.3733, 'lon': -81.3627},

        # Costa Rica
        "San José": {'lat': 9.93, 'lon': -84.08},
        "Liberia": {'lat': 10.63, 'lon': -85.44},

        # Panamá
        "Ciudad de Panamá": {'lat': 9.01, 'lon': -79.52},
        "David": {'lat': 8.43, 'lon': -82.43},
        "Santiago de Veraguas": {'lat': 8.10, 'lon': -80.97},

        # Belice
        "Belmopán": {'lat': 17.25, 'lon': -88.77}
    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        ax.text(
            lon_ci, lat_ci - 0.08, nombre,
            fontsize=7.4, color='#2C3E50', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                PathEffects.Normal()
            ]
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.198, 0.25, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f"Logo no encontrado en {DEFAULT_LOGO_PATH}")

    # Crédito
    ax.text(0.8, 0.4, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)

    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)

    ax.text(0.25, 0.02, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/centroamerica_mapa_precipitacion_multimodelo.png",
                dpi=800, bbox_inches='tight')
    print(f"✅ Mapa guardado: {output_dir}/centroamerica_mapa_precipitacion_multimodelo.png")

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax


# --- USO ---
# Asegúrate de tener cargado gdf_estados y ds_multimodelo de tu clase MultiModeloMeteorologico
fig, ax = graficar_precipitacion_centroamerica(ds_multi, gdf_estados=gdf_estados)


#-----LA ESPAÑOLA (HAITÍ Y REPÚBLICA DOMINICANA) - PRECIPITACIÓN MULTIMODELO----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_hispaniola(precip_dataarray, gdf_estados=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para La Española (Haití y República Dominicana).
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    La precipitación se muestra tanto sobre tierra como sobre el mar.

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con todos los estados
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """

    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # --- FILTRAR HAITÍ Y REPÚBLICA DOMINICANA ---
    haiti = gdf_estados[gdf_estados['admin'] == 'Haiti'].copy()
    republica_dominicana = gdf_estados[gdf_estados['admin'] == 'Dominican Republic'].copy()
    hispaniola = pd.concat([haiti, republica_dominicana])

    # --- PARÁMETROS DE ÁREA ---
    lon_min_deg = -76
    lon_max_deg = -67
    lat_min = 16
    lat_max = 20.8

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f" LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f" LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f" Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_max+1, lat_min),
            longitude=slice(lon_min_deg, lon_max_deg)
        )
    else:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_min, lat_max+1),
            longitude=slice(lon_min_deg, lon_max_deg)
        )

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f" LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f" LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f" Precipitación: Min={precip_vals.min():.2f}, Max={precip_vals.max():.2f}, Mean={precip_vals.mean():.2f} mm")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)

    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA SOLO POR UMBRAL (sin restringir a tierra) ---
    umbral_precip = 3
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_masked_fine)

    if precip_max < 50:
        vmin_colorbar, vmax_colorbar = 1, 50
    elif 50 <= precip_max < 100:
        vmin_colorbar, vmax_colorbar = 0, 100
    elif 100 <= precip_max < 150:
        vmin_colorbar, vmax_colorbar = 0, 150
    else:
        vmin_colorbar, vmax_colorbar = 1, 200

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)

    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN (SOBRE TIERRA Y MAR) ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )

    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTAS CON SOMBRA 3D ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- FRONTERA INTERNACIONAL ENTRE HAITÍ Y RD ---
    frontera_internacional = haiti.unary_union.intersection(republica_dominicana.unary_union)
    ax.add_geometries([frontera_internacional], crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries([frontera_internacional], crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- PROVINCIAS INTERNAS ---
    # Provincias de Haití
    provincias_haiti = gdf_estados[(gdf_estados['admin'] == 'Haiti') & (gdf_estados['type'] != 'Country')]
    provincias_haiti.boundary.plot(ax=ax, edgecolor='black', linewidth=0.3, zorder=4, transform=ccrs.PlateCarree())

    # Provincias de República Dominicana
    provincias_rd = gdf_estados[(gdf_estados['admin'] == 'Dominican Republic') & (gdf_estados['type'] != 'Country')]
    provincias_rd.boundary.plot(ax=ax, edgecolor='black', linewidth=0.3, zorder=4, transform=ccrs.PlateCarree())

    # --- BARRA DE COLOR ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.015, 0.088, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(mm)", fontsize=11, weight='bold')

    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])

    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)

    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES PRINCIPALES DE LA ESPAÑOLA ---
    ciudades_principales = {
        "Puerto Príncipe": {'lat': 18.54, 'lon': -72.34},
        "Santiago": {'lat': 19.45, 'lon': -70.70},
        "Santo Domingo": {'lat': 18.48, 'lon': -69.90},
        "Cotuí": {'lat': 19.08, 'lon': -70.16},
        "San Juan": {'lat': 18.80, 'lon': -71.22},
        "Barahona": {'lat': 18.20, 'lon': -71.10},
        "Punta Cana": {'lat': 18.58, 'lon': -68.40},
        "Jérémie": {'lat': 18.64, 'lon': -74.12},
        "La Romana": {'lat': 18.43, 'lon': -68.97},
    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        ax.text(
            lon_ci, lat_ci - 0.08, nombre,
            fontsize=7.4, color='#2C3E50', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                PathEffects.Normal()
            ]
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.198, 0.19, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f"Logo no encontrado en {DEFAULT_LOGO_PATH}")

    # Crédito
    ax.text(0.8, 0.35, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)

    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)

    ax.text(0.25, 0.02, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/hispaniola_mapa_precipitacion_multimodelo.png",
                dpi=800, bbox_inches='tight')
    print(f"✅ Mapa guardado: {output_dir}/hispaniola_mapa_precipitacion_multimodelo.png")

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax


# --- USO ---
# Asegúrate de tener cargado gdf_estados y ds_multimodelo de tu clase MultiModeloMeteorologico
fig, ax = graficar_precipitacion_hispaniola(ds_multi, gdf_estados=gdf_estados)


#-----ANTILLAS MENORES - PRECIPITACIÓN MULTIMODELO----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_antillas_menores(precip_dataarray, gdf_estados=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para las Antillas Menores y guarda la figura en outputs/maps.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    """
    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # Filtrar países de las Antillas Menores
    paises = ['Antigua and Barbuda', 'Dominica', 'Saint Lucia', 'Saint Vincent and the Grenadines',
              'Grenada', 'Barbados', 'Trinidad and Tobago', 'Saint Kitts and Nevis',
              'Martinique', 'Guadeloupe', 'Montserrat', 'Anguilla',
              'British Virgin Islands', 'United States Virgin Islands', 'Aruba',
              'Curaçao', 'Sint Maarten', 'Saint Barthélemy', 'Bonaire',
              'France']  # Agregar Francia para territorios franceses

    antillas_menores = gdf_estados[gdf_estados['admin'].isin(paises)].copy()

    # --- PARÁMETROS DE ÁREA ---
    lon_min_deg = -71
    lon_max_deg = -57
    lat_min = 9.5
    lat_max = 19

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f"  LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f"  LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f"  Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max, lat_min),
                                           longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max),
                                           longitude=slice(lon_min_deg, lon_max_deg))

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f"  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f"  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f"  Precipitación: Min={precip_vals.min():.2f}, Max={precip_vals.max():.2f}, Mean={precip_vals.mean():.2f} mm")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA ---
    umbral_precip = 3
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_fine_smooth)
    print(f"Precipitación máxima: {precip_max:.2f} mm")

    if precip_max < 50:
        vmin_colorbar, vmax_colorbar = 1, 50
    elif 50 <= precip_max < 100:
        vmin_colorbar, vmax_colorbar = 0, 100
    else:
        vmin_colorbar, vmax_colorbar = 0, 150

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    # No dibujar fronteras internas, solo costas

    # --- COLORES ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTA CON SOMBRA ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- BARRA DE COLOR ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.015, 0.35, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(mm)", fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES ---
    ciudades_principales = {
        # Islas Vírgenes
        "Islas Vírgenes Británicas": {'lat': 18.43, 'lon': -64.62},

        # Islas de Sotavento (Norte)
        "Anguila": {'lat': 18.22, 'lon': -63.05},

        # Saint Kitts y Nevis
        "San Cristóbal": {'lat': 17.30, 'lon': -62.72},

        # Antigua y Barbuda
        "Antigua": {'lat': 17.12, 'lon': -61.85},

        # Guadeloupe
        "Guadalupe": {'lat': 16.24, 'lon': -61.53},

        # Dominica
        "Dominica": {'lat': 15.30, 'lon': -61.39},

        # Martinique
        "Martinica": {'lat': 14.61, 'lon': -61.08},

        # Saint Lucia
        "Santa Lucía": {'lat': 14.01, 'lon': -60.99},

        # Saint Vincent y Grenadinas
        "San Vicente": {'lat': 13.1633, 'lon': -61.2233},

        # Barbados
        "Barbados": {'lat': 13.10, 'lon': -59.61},

        # Grenada
        "Granada": {'lat': 12.05, 'lon': -61.75},

        # Trinidad y Tobago
        "Puerto España": {'lat': 10.6702, 'lon': -61.5038},
        "Tobago": {'lat': 11.18, 'lon': -60.74},

        # Islas ABC (Antillas Neerlandesas)
        "Aruba": {'lat': 12.52, 'lon': -70.03},
        "Curazao": {'lat': 12.11, 'lon': -68.93}
    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        ax.text(
            lon_ci, lat_ci - 0.08, nombre,
            fontsize=7.4, color='#2C3E50', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                PathEffects.Normal()
            ]
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.55, 0.56, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print("Logo no encontrado")

    # Crédito
    ax.text(0.5, 0.7, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/antillas_menores_mapa_precipitacion_multimodelo.png", dpi=800, bbox_inches='tight')

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax

# --- USO ---
print("🌧️ Generando mapa de PRECIPITACIÓN para las Antillas Menores...")
fig, ax = graficar_precipitacion_dataarray(ds_multi, gdf_estados=gdf_estados)


#----PUERTO RICO - PRECIPITACIÓN MULTIMODELO (EN PULGADAS)----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_puerto_rico(precip_dataarray, gdf_estados=None, municipios_pr=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para Puerto Rico.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    La precipitación se muestra tanto sobre tierra como sobre el mar.
    (Conversión de mm a pulgadas)

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con todos los estados
    municipios_pr : GeoDataFrame
        GeoDataFrame con municipios de Puerto Rico (opcional)
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """

    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # --- FILTRAR PUERTO RICO ---
    puerto_rico = gdf_estados[gdf_estados['admin'] == 'Puerto Rico'].copy()

    # --- PARÁMETROS DE ÁREA (PUERTO RICO) ---
    lon_min_deg = -68.0
    lon_max_deg = -65
    lat_min = 17.25
    lat_max = 19.0

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f" LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f" LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f" Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min_deg, lon_max_deg)
        )
    else:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_min, lat_max+1),
            longitude=slice(lon_min_deg, lon_max_deg)
        )

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f" LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f" LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values

    # --- CONVERTIR DE mm A PULGADAS ---
    precip_vals = precip_vals / 25.4

    print(f" Precipitación (en pulgadas): Min={precip_vals.min():.2f}, Max={precip_vals.max():.2f}, Mean={precip_vals.mean():.2f}")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)

    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA SOLO POR UMBRAL (sin restringir a tierra) ---
    umbral_precip = 0.1  #
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO (en pulgadas) ---
    precip_max = np.nanmax(precip_masked_fine)

    if precip_max < 2:
        vmin_colorbar, vmax_colorbar = 0, 2
    elif 2 <= precip_max < 4:
        vmin_colorbar, vmax_colorbar = 0, 4
    else:
        vmin_colorbar, vmax_colorbar = 0, 6

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)

    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5), 2))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )

    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%.1f', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTAS CON SOMBRA ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- FRONTERA DE PUERTO RICO ---
    puerto_rico.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=5, transform=ccrs.PlateCarree())

    # --- MUNICIPIOS (si están disponibles) ---
    if municipios_pr is not None:
        municipios_pr.boundary.plot(ax=ax, edgecolor='grey', linewidth=0.2, alpha=0.8, zorder=8, transform=ccrs.PlateCarree())

    # --- BARRA DE COLOR ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.015, 0.088, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(
    cf, cax=cax, orientation='horizontal',
    ticks=np.arange(vmin_colorbar, vmax_colorbar + 0.5, 0.5)
)

    cbar.set_label("(pulgadas)", fontsize=11, weight='bold')

    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])

    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)

    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES PRINCIPALES DE PUERTO RICO ---
    ciudades_principales = {
        "San Juan": {'lat': 18.46, 'lon': -66.105},
        "Ponce": {'lat': 18.01, 'lon': -66.61},
        "Caguas": {'lat': 18.24, 'lon': -66.04},
        "Mayagüez": {'lat': 18.20, 'lon': -67.14},
        "Arecibo": {'lat': 18.47, 'lon': -66.72},
        "Fajardo": {'lat': 18.33, 'lon': -65.65},
        "Aguadilla": {'lat': 18.43, 'lon': -67.15},
        "Humacao": {'lat': 18.15, 'lon': -65.83},
        "Utuado": {'lat': 18.27, 'lon': -66.70},
        "Guayama": {'lat': 17.98, 'lon': -66.11},
        "Vieques": {'lat': 18.12, 'lon': -65.44},
        "Culebra": {'lat': 18.32, 'lon': -65.29},
    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        ax.text(
            lon_ci, lat_ci - 0.02, nombre,
            fontsize=7.4, color='#2C3E50', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                PathEffects.Normal()
            ]
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.198, 0.2, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f"Logo no encontrado en {DEFAULT_LOGO_PATH}")

    # Crédito
    ax.text(0.5, 0.34, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)

    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)

    ax.text(0.25, 0.02, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/puerto_rico_mapa_precipitacion_multimodelo_pulgadas.png",
                dpi=800, bbox_inches='tight')
    print(f"✅ Mapa guardado: {output_dir}/puerto_rico_mapa_precipitacion_multimodelo_pulgadas.png")

    plt.show()

    return fig, ax


# --- CARGAR MUNICIPIOS (OPCIONAL) ---
try:
    municipios_pr = gpd.read_file(DEFAULT_US_COUNTY_SHAPEFILE)
    municipios_pr = municipios_pr[municipios_pr['STATEFP'] == '72']
    print("✅ Municipios de Puerto Rico cargados")
except:
    print("⚠️ Municipios no encontrados, se continuará sin ellos")
    municipios_pr = None

# --- USO ---
# Asegúrate de tener cargado gdf_estados y ds_multimodelo de tu clase MultiModeloMeteorologico
fig, ax = graficar_precipitacion_puerto_rico(ds_multi, gdf_estados=gdf_estados, municipios_pr=municipios_pr)


#-----COLOMBIA Y VENEZUELA - PRECIPITACIÓN MULTIMODELO----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_colombia_venezuela(precip_dataarray, gdf_estados=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para Colombia y Venezuela.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    La precipitación se muestra tanto sobre tierra como sobre el mar.

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con todos los estados
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """

    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # Filtrar países de Colombia y Venezuela
    paises = ['Colombia', 'Venezuela']
    gdf_cv = gdf_estados[gdf_estados['admin'].isin(paises)].copy()

    # --- PARÁMETROS DE ÁREA ---
    lon_min_deg = -80
    lon_max_deg = -58
    lat_min = 0.5
    lat_max = 13

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f" LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f" LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f" Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_max+1, lat_min),
            longitude=slice(lon_min_deg, lon_max_deg)
        )
    else:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_min, lat_max+1),
            longitude=slice(lon_min_deg, lon_max_deg)
        )

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f" LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f" LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f" Precipitación: Min={precip_vals.min():.2f}, Max={precip_vals.max():.2f}, Mean={precip_vals.mean():.2f} mm")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)

    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA SOLO POR UMBRAL (sin restringir a tierra) ---
    umbral_precip = 5
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_masked_fine)

    if precip_max < 50:
        vmin_colorbar, vmax_colorbar = 1, 50
    elif 50 <= precip_max < 100:
        vmin_colorbar, vmax_colorbar = 0, 100
    elif 100 <= precip_max < 150:
        vmin_colorbar, vmax_colorbar = 0, 200
    else:
        vmin_colorbar, vmax_colorbar = 1, 300

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)

    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN (SOBRE TIERRA Y MAR) ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )

    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTAS CON SOMBRA 3D ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- FRONTERAS PAÍSES CON EFECTO 3D ---
    for pais in paises:
        gdf_pais = gdf_cv[gdf_cv['admin'] == pais]
        if len(gdf_pais) > 0:
            fronteras = gdf_pais.unary_union.boundary
            ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                              edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                              path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                              zorder=6)
            ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                              edgecolor='white', facecolor='none', linewidth=1.3,
                              path_effects=[PathEffects.Normal()], zorder=7)

    # --- PROVINCIAS INTERNAS ---
    for pais in paises:
        provincias = gdf_cv[(gdf_cv['admin'] == pais) & (gdf_cv['type'] != 'Country')]
        if len(provincias) > 0:
            provincias.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35,
                                      zorder=4, transform=ccrs.PlateCarree())

    # --- BARRA DE COLOR (VERTICAL, ESQUINA INFERIOR IZQUIERDA) ---
    cax = inset_axes(ax, width="2.5%", height="45%", loc='lower left',
                     bbox_to_anchor=(0.02, 0.15, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label("(mm)", fontsize=10, weight='bold')

    label_obj = cbar.ax.yaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])

    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES PRINCIPALES DE COLOMBIA Y VENEZUELA ---
    ciudades_principales = {
        # Colombia
        "Bogotá": {'lat': 4.60971, 'lon': -74.08175},
        "Medellín": {'lat': 6.2442, 'lon': -75.5812},
        "Cali": {'lat': 3.4516, 'lon': -76.5320},
        "Barranquilla": {'lat': 10.9685, 'lon': -74.7813},
        "Montería": {'lat': 8.7471, 'lon': -75.8894},
        "Pereira": {'lat': 4.8143, 'lon': -75.6946},
        "Cúcuta": {'lat': 7.89, 'lon': -72.4963},
        "Cartagena": {'lat': 10.395, 'lon': -75.4833},
        "Valledupar": {'lat': 10.4719, 'lon': -73.2527},
        "Maicao": {'lat': 11.3775, 'lon': -72.2383},
        "Bucaramanga": {'lat': 7.1238, 'lon': -73.1216},
        # Venezuela
        "Caracas": {'lat': 10.4880, 'lon': -66.8792},
        "Maracaibo": {'lat': 10.6539, 'lon': -71.64597},
        "Valencia": {'lat': 10.1620, 'lon': -68.0077},
        "Maturín": {'lat': 9.7500, 'lon': -63.1800},
        "Barcelona": {'lat': 10.1363, 'lon': -64.6862},
        "Ciudad Bolívar": {'lat': 8.0833, 'lon': -63.6000},
        "Mérida": {'lat': 8.57, 'lon': -71.18},
        "Barquisimeto": {'lat': 10.0683, 'lon': -69.3452},
        "Coro": {'lat': 11.3980, 'lon': -69.6794},
        "Cd Guayana": {'lat': 8.3525, 'lon': -62.6430},
    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        ax.text(
            lon_ci, lat_ci - 0.08, nombre,
            fontsize=7.4, color='#2C3E50', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                PathEffects.Normal()
            ]
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.12, 0.01, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f"Logo no encontrado en {DEFAULT_LOGO_PATH}")

    # Crédito
    ax.text(0.9, 0.15, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)

    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)

    ax.text(0.25, 0.02, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/colombia_venezuela_mapa_precipitacion_multimodelo.png",
                dpi=800, bbox_inches='tight')
    print(f"✅ Mapa guardado: {output_dir}/colombia_venezuela_mapa_precipitacion_multimodelo.png")

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax


# --- USO ---
# Asegúrate de tener cargado gdf_estados y ds_multimodelo de tu clase MultiModeloMeteorologico
fig, ax = graficar_precipitacion_colombia_venezuela(ds_multi, gdf_estados=gdf_estados)


#-----ESTADOS UNIDOS - PRECIPITACIÓN MULTIMODELO----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_estados_unidos(precip_dataarray, gdf_estados=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para Estados Unidos.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    La precipitación se muestra tanto sobre tierra como sobre el mar.

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con todos los estados
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """

    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # Filtrar Estados Unidos
    paises = ['United States of America']
    gdf_usa = gdf_estados[gdf_estados['admin'].isin(paises)].copy()

    # --- PARÁMETROS DE ÁREA (ESTADOS UNIDOS CONTINENTAL) ---
    lon_min_deg = -125
    lon_max_deg = -66
    lat_min = 24
    lat_max = 50

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f" LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f" LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f" Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_max+1, lat_min),
            longitude=slice(lon_min_deg, lon_max_deg)
        )
    else:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_min, lat_max+1),
            longitude=slice(lon_min_deg, lon_max_deg)
        )

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f" LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f" LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values

    # --- CONVERTIR DE MILÍMETROS A PULGADAS ---
    precip_vals_inches = precip_vals / 25.4

    print(f" Precipitación: Min={precip_vals_inches.min():.2f}, Max={precip_vals_inches.max():.2f}, Mean={precip_vals_inches.mean():.2f} in")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)

    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals_inches.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA SOLO POR UMBRAL (sin restringir a tierra) ---
    umbral_precip = 0.1      # 0.2 pulgadas (equivalente a 5 mm)
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_masked_fine)

    if precip_max < 2:
        vmin_colorbar, vmax_colorbar = 0, 2
    elif 2 <= precip_max < 4:
        vmin_colorbar, vmax_colorbar = 0, 4
    else:
        vmin_colorbar, vmax_colorbar = 0, 8

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)

    # AGREGAR LAGOS
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN (SOBRE TIERRA Y MAR) ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5), 1))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )

    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%.1f', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTAS CON SOMBRA 3D ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- FRONTERAS PAÍSES CON EFECTO 3D ---
    for pais in paises:
        gdf_pais = gdf_usa[gdf_usa['admin'] == pais]
        if len(gdf_pais) > 0:
            fronteras = gdf_pais.unary_union.boundary
            ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                              edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                              path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                              zorder=6)
            ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                              edgecolor='white', facecolor='none', linewidth=1.3,
                              path_effects=[PathEffects.Normal()], zorder=7)

    # --- ESTADOS INTERNOS ---
    for pais in paises:
        estados = gdf_usa[(gdf_usa['admin'] == pais) & (gdf_usa['type'] != 'Country')]
        if len(estados) > 0:
            estados.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35,
                                      zorder=4, transform=ccrs.PlateCarree())

    # --- BARRA DE COLOR (HORIZONTAL) ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.05, 0.001, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(pulgadas)", fontsize=11, weight='bold')

    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])

    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)

    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES PRINCIPALES DE ESTADOS UNIDOS ---
    ciudades_principales = {
        # Costa Oeste
        "Los Angeles": {'lat': 34.05, 'lon': -118.24},
        "San Francisco": {'lat': 37.77, 'lon': -122.42},
        "Seattle": {'lat': 47.61, 'lon': -122.33},
        "Portland": {'lat': 45.52, 'lon': -122.68},
        "San Diego": {'lat': 32.72, 'lon': -117.16},
        "Las Vegas": {'lat': 36.17, 'lon': -115.14},
        "Reno": {'lat': 39.5305, 'lon': -119.8130},
        # Suroeste
        "Phoenix": {'lat': 33.45, 'lon': -112.07},
        "Tucson": {'lat': 32.2566, 'lon': -110.9697},
        "Denver": {'lat': 39.74, 'lon': -104.99},
        "Albuquerque": {'lat': 35.08, 'lon': -106.65},
        "Salt Lake City": {'lat': 40.76, 'lon': -111.89},
        # Texas
        "Houston": {'lat': 29.76, 'lon': -95.37},
        "Dallas": {'lat': 32.78, 'lon': -96.80},
        "San Antonio": {'lat': 29.42, 'lon': -98.49},
        "McAllen": {'lat': 26.2066, 'lon': -98.2325},
        "Amarillo": {'lat': 35.205, 'lon': -101.8369},
        # Oklahoma
        "Oklahoma City": {'lat': 35.4738, 'lon': -97.5205},
        # Sureste
        "Miami": {'lat': 25.76, 'lon': -80.27},
        "Orlando": {'lat': 28.5308, 'lon': -81.3825},
        "Jacksonville": {'lat': 30.3316, 'lon': -81.6577},
        "Atlanta": {'lat': 33.75, 'lon': -84.39},
        "New Orleans": {'lat': 29.95, 'lon': -90.07},
        "Nashville": {'lat': 36.16, 'lon': -86.78},
        "Charlotte": {'lat': 35.23, 'lon': -80.84},
        # Costa Este
        "New York": {'lat': 40.71, 'lon': -74.01},
        "Boston": {'lat': 42.36, 'lon': -71.06},
        "Washington DC": {'lat': 38.91, 'lon': -77.04},
        # Medio Oeste
        "Chicago": {'lat': 41.88, 'lon': -87.63},
        "Detroit": {'lat': 42.33, 'lon': -83.1},
        "Minneapolis": {'lat': 44.98, 'lon': -93.27},
        "Kansas City": {'lat': 39.10, 'lon': -94.58},
        "Wichita": {'lat': 37.6916, 'lon': -97.3286},
        "Omaha": {'lat': 41.2591, 'lon': -95.9347},
        "Memphis": {'lat': 35.1402, 'lon': -90.0355},
        "Cincinati": {'lat': 39.1058, 'lon': -84.5141},
        "St. Louis": {'lat': 38.63, 'lon': -90.20}
    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        ax.text(
            lon_ci, lat_ci - 0.12, nombre,
            fontsize=7.4, color='#2C3E50', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                PathEffects.Normal()
            ]
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.0, 0.12, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f"Logo no encontrado en {DEFAULT_LOGO_PATH}")

    # Crédito
    ax.text(0.97, 0.1, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)

    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)

    ax.text(0.25, 0.02, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/estados_unidos_mapa_precipitacion_multimodelo.png",
                dpi=800, bbox_inches='tight')
    print(f"✅ Mapa guardado: {output_dir}/estados_unidos_mapa_precipitacion_multimodelo.png")

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax


# --- USO ---
# Asegúrate de tener cargado gdf_estados y ds_multimodelo de tu clase MultiModeloMeteorologico
fig, ax = graficar_precipitacion_estados_unidos(ds_multi, gdf_estados=gdf_estados)


#-----FLORIDA - PRECIPITACIÓN MULTIMODELO----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_florida(precip_dataarray, gdf_estados=None, condados_florida=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para Florida en PULGADAS.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    La precipitación se muestra tanto sobre tierra como sobre el mar.

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con todos los estados
    condados_florida : GeoDataFrame
        GeoDataFrame con condados de Florida (opcional)
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """

    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # Filtrar Florida
    gdf_florida = gdf_estados[
        (gdf_estados['admin'] == 'United States of America') &
        (gdf_estados['name'] == 'Florida')
    ].copy()

    # --- PARÁMETROS DE ÁREA (FLORIDA) ---
    lon_min_deg = -88.5
    lon_max_deg = -77.5
    lat_min = 24.3
    lat_max = 31.5

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f" LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f" LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f" Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA (EXTENDER 1 GRADO HACIA ABAJO) ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_max+1, lat_min-1),  # Extender 1 grado hacia abajo
            longitude=slice(lon_min_deg, lon_max_deg)
        )
    else:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_min-1, lat_max+1),  # Extender 1 grado hacia abajo
            longitude=slice(lon_min_deg, lon_max_deg)
        )

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f" LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f" LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values

    # --- CONVERTIR DE MILÍMETROS A PULGADAS ---
    precip_vals_inches = precip_vals / 25.4

    print(f" Precipitación: Min={precip_vals_inches.min():.2f}, Max={precip_vals_inches.max():.2f}, Mean={precip_vals_inches.mean():.2f} in")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)

    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals_inches.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA SOLO POR UMBRAL (sin restringir a tierra) ---
    umbral_precip = 0.1  # 0.2 pulgadas (equivalente a ~5mm)
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO EN PULGADAS ---
    precip_max = np.nanmax(precip_masked_fine)

    if precip_max < 2:
        vmin_colorbar, vmax_colorbar = 0, 2
    elif 2 <= precip_max < 4:
        vmin_colorbar, vmax_colorbar = 0, 4
    else:
        vmin_colorbar, vmax_colorbar = 0, 8

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)

    # AGREGAR LAGOS
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN (SOBRE TIERRA Y MAR) ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5), 1))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )

    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%.1f', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTAS CON SOMBRA 3D ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- FRONTERA DE FLORIDA CON EFECTO 3D ---
    if len(gdf_florida) > 0:
        fronteras = gdf_florida.unary_union.boundary
        ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                          edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                          path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                          zorder=6)
        ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                          edgecolor='white', facecolor='none', linewidth=1.3,
                          path_effects=[PathEffects.Normal()], zorder=7)

    # --- CONDADOS DE FLORIDA ---
    if condados_florida is not None:
        print(f"✅ Dibujando {len(condados_florida)} condados de Florida")
        condados_florida.boundary.plot(ax=ax, edgecolor='grey', linewidth=0.3, alpha=0.8,
                                       zorder=8, transform=ccrs.PlateCarree())
    else:
        print("⚠️ No se proporcionaron condados de Florida")

    # --- BARRA DE COLOR (HORIZONTAL) CON TICKS CADA 0.5 ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.02, 0.05, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(in)", fontsize=11, weight='bold')

    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])

    # Configurar ticks cada 0.5 pulgadas
    tick_values = np.arange(0, vmax_colorbar + 0.5, 1)
    cbar.set_ticks(tick_values)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)

    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES PRINCIPALES DE FLORIDA ---
    ciudades_principales = {
        "Miami": {'lat': 25.76, 'lon': -80.19},
        "Tampa": {'lat': 27.95, 'lon': -82.46},
        "Orlando": {'lat': 28.54, 'lon': -81.38},
        "Jacksonville": {'lat': 30.33, 'lon': -81.66},
        "Tallahassee": {'lat': 30.44, 'lon': -84.28},
        "Fort Lauderdale": {'lat': 26.12, 'lon': -80.14},
        "West Palm Beach": {'lat': 26.71, 'lon': -80.05},
        "Naples": {'lat': 26.14, 'lon': -81.79},
        "Fort Myers": {'lat': 26.64, 'lon': -81.87},
        "Sarasota": {'lat': 27.34, 'lon': -82.53},
        "Pensacola": {'lat': 30.42, 'lon': -87.22},
        "Gainesville": {'lat': 29.65, 'lon': -82.32},
        "Daytona Beach": {'lat': 29.21, 'lon': -81.02},
        "Key West": {'lat': 24.56, 'lon': -81.78},
        "Panama City": {'lat': 30.16, 'lon': -85.66},
    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        # BANNER AZUL AJUSTADO PARA EL NOMBRE DE LA CIUDAD
        ax.text(
            lon_ci, lat_ci - 0.08, nombre,
            fontsize=7.4, color='white', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#1a5490',
                      edgecolor='white', linewidth=0.8, alpha=0.95)
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.07, 0.05, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f"Logo no encontrado en {DEFAULT_LOGO_PATH}")

    # Crédito
    ax.text(0.90, 0.01, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)

    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)

    ax.text(0.25, 0.02, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/florida_mapa_precipitacion_multimodelo.png",
                dpi=800, bbox_inches='tight')
    print(f"✅ Mapa guardado: {output_dir}/florida_mapa_precipitacion_multimodelo.png")

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax


# --- CARGAR CONDADOS DE FLORIDA ---
try:
    condados_florida = gpd.read_file(DEFAULT_US_COUNTY_SHAPEFILE)
    condados_florida = condados_florida[condados_florida['STATEFP'] == '12']  # FIPS code 12 = Florida
    print(f"✅ {len(condados_florida)} condados de Florida cargados")
except:
    print("⚠️ Condados no encontrados, se continuará sin ellos")
    condados_florida = None

# --- USO ---
# Asegúrate de tener cargado gdf_estados y ds_multimodelo de tu clase MultiModeloMeteorologico
fig, ax = graficar_precipitacion_florida(ds_multi, gdf_estados=gdf_estados, condados_florida=condados_florida)


#-----TEXAS - PRECIPITACIÓN MULTIMODELO----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_texas(precip_dataarray, gdf_estados=None, condados_texas=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para Texas en PULGADAS.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    La precipitación se muestra tanto sobre tierra como sobre el mar.

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con todos los estados
    condados_texas : GeoDataFrame
        GeoDataFrame con condados de Texas (opcional)
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """
    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # Filtrar Texas
    gdf_texas = gdf_estados[
        (gdf_estados['admin'] == 'United States of America') &
        (gdf_estados['name'] == 'Texas')
    ].copy()

    # --- PARÁMETROS DE ÁREA (TEXAS) ---
    lon_min_deg = -108
    lon_max_deg = -92
    lat_min = 25.5
    lat_max = 37

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f"  LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f"  LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f"  Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA (EXTENDER 1 GRADO) ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_max+1, lat_min-1),
            longitude=slice(lon_min_deg, lon_max_deg)
        )
    else:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_min-1, lat_max+1),
            longitude=slice(lon_min_deg, lon_max_deg)
        )

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f"  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f"  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values

    # --- CONVERTIR DE MILÍMETROS A PULGADAS ---
    precip_vals_inches = precip_vals / 25.4
    print(f"  Precipitación: Min={precip_vals_inches.min():.2f}, Max={precip_vals_inches.max():.2f}, Mean={precip_vals_inches.mean():.2f} in")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 500)
    lat_fine = np.linspace(lat.min(), lat.max(), 500)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals_inches.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA SOLO POR UMBRAL (sin restringir a tierra) ---
    umbral_precip = 0.1  # 0.1 pulgadas
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO EN PULGADAS ---
    precip_max = np.nanmax(precip_masked_fine)
    print(f"Precipitación máxima: {precip_max:.2f} in")

    if precip_max < 2:
        vmin_colorbar, vmax_colorbar = 0, 2
    elif 2 <= precip_max < 4:
        vmin_colorbar, vmax_colorbar = 0, 4
    else:
        vmin_colorbar, vmax_colorbar = 0, 8

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(18, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)

    # AGREGAR LAGOS
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN (SOBRE TIERRA Y MAR) ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5), 1))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%.1f', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTAS CON SOMBRA 3D ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- FRONTERA DE TEXAS CON EFECTO 3D ---
    if len(gdf_texas) > 0:
        fronteras = gdf_texas.unary_union.boundary
        ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                          edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                          path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                          zorder=6)
        ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                          edgecolor='white', facecolor='none', linewidth=1.3,
                          path_effects=[PathEffects.Normal()], zorder=7)

    # --- CONDADOS DE TEXAS ---
    if condados_texas is not None:
        print(f"✅ Dibujando {len(condados_texas)} condados de Texas")
        condados_texas.boundary.plot(ax=ax, edgecolor='grey', linewidth=0.3, alpha=0.8,
                                       zorder=8, transform=ccrs.PlateCarree())
    else:
        print("⚠️ No se proporcionaron condados de Texas")

    # --- BARRA DE COLOR (HORIZONTAL) CON TICKS CADA 0.5 ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.02, 0.05, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(in)", fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])
    # Configurar ticks cada 1 pulgada
    tick_values = np.arange(0, vmax_colorbar + 0.5, 1)
    cbar.set_ticks(tick_values)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES PRINCIPALES DE TEXAS ---
    ciudades_principales = {
        "Houston": {'lat': 29.76, 'lon': -95.37},
        "San Antonio": {'lat': 29.42, 'lon': -98.49},
        "Dallas": {'lat': 32.78, 'lon': -96.80},
        "Austin": {'lat': 30.27, 'lon': -97.74},
        "El Paso": {'lat': 31.76, 'lon': -106.49},
        "Corpus Christi": {'lat': 27.80, 'lon': -97.40},
        "Laredo": {'lat': 27.51, 'lon': -99.51},
        "Lubbock": {'lat': 33.58, 'lon': -101.86},
        "Amarillo": {'lat': 35.22, 'lon': -101.83},
        "Brownsville": {'lat': 25.90, 'lon': -97.50},
        "Odessa": {'lat': 31.85, 'lon': -102.37},
        "Beaumont": {'lat': 30.09, 'lon': -94.13},
        "Waco": {'lat': 31.55, 'lon': -97.15},
    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        # BANNER AZUL AJUSTADO PARA EL NOMBRE DE LA CIUDAD
        ax.text(
            lon_ci, lat_ci - 0.14, nombre,
            fontsize=9, color='white', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#1a5490',
                      edgecolor='white', linewidth=0.8, alpha=0.95)
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.07, 0.05, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f"Logo no encontrado en {DEFAULT_LOGO_PATH}")

    # Crédito
    ax.text(0.90, 0.01, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/texas_mapa_precipitacion_multimodelo.png",
                dpi=800, bbox_inches='tight')
    print(f"✅ Mapa guardado: {output_dir}/texas_mapa_precipitacion_multimodelo.png")

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax

# --- CARGAR CONDADOS DE TEXAS ---
try:
    condados_texas = gpd.read_file(DEFAULT_US_COUNTY_SHAPEFILE)
    condados_texas = condados_texas[condados_texas['STATEFP'] == '48']  # FIPS code 48 = Texas
    print(f"✅ {len(condados_texas)} condados de Texas cargados")
except:
    print("⚠️ Condados no encontrados, se continuará sin ellos")
    condados_texas = None

# --- USO ---
print("🌧️ Generando mapa de PRECIPITACIÓN para Texas...")
fig, ax = graficar_precipitacion_texas(ds_multi, gdf_estados=gdf_estados, condados_texas=condados_texas)


#-----MÉXICO - PRECIPITACIÓN----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString
from scipy.ndimage import gaussian_filter

def graficar_precipitacion_mexico(precip_dataarray, gdf_estados=None):
    """
    Grafica precipitación acumulada para México en PULGADAS.

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con todos los estados
    """

    # --- VALIDACIÓN DE PARÁMETROS ---
    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # --- PARÁMETROS DE ÁREA ---
    lon_min_deg = -119
    lon_max_deg = -86
    lat_min = 14
    lat_max = 33

    # --- FILTRAR MÉXICO ---
    mexico = gdf_estados[gdf_estados['admin'] == 'Mexico'].copy()

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER Y CORREGIR COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    precip_corrected = precip_dataarray.assign_coords(longitude=lon_corrected)

    # --- SELECCIONAR ÁREA ---
    # --- SELECCIONAR ÁREA (con margen adicional SOLO para los datos) ---
    margen_lon = 2  # grados extra a cada lado en longitud
    margen_lat = 2  # grados extra a cada lado en latitud

    if lat_orig[0] > lat_orig[-1]:
      precip_area = precip_corrected.sel(
      latitude=slice(lat_max + margen_lat, lat_min - margen_lat),
      longitude=slice(lon_min_deg - margen_lon, lon_max_deg + margen_lon)
    )
    else:
      precip_area = precip_corrected.sel(
      latitude=slice(lat_min - margen_lat, lat_max + margen_lat),
      longitude=slice(lon_min_deg - margen_lon, lon_max_deg + margen_lon)
    )


    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    if precip_area.values.ndim == 3:
        precip_vals = precip_area.values[0, :, :]
    else:
        precip_vals = precip_area.values

    # --- CONVERTIR DE MILÍMETROS A PULGADAS ---
    precip_vals_inches = precip_vals

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)

    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals_inches.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- APLICAR MÁSCARA SOLO POR UMBRAL (sin restringir a tierra) ---
    umbral_precip = 1  # mm
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_masked_fine)

    if precip_max < 50:
        vmin_colorbar, vmax_colorbar = 1, 50
    elif 50 <= precip_max < 100:
        vmin_colorbar, vmax_colorbar = 0, 100
    elif 100 <= precip_max < 150:
        vmin_colorbar, vmax_colorbar = 0, 200
    else:
        vmin_colorbar, vmax_colorbar = 1, 300

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- PLOTEO ---
    fig = plt.figure(figsize=(15, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    buffer_lat = 1
    ax.set_extent([lon_min_deg, lon_max_deg + buffer_lat, lat_min - buffer_lat, lat_max + buffer_lat],
                  crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')

    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels,
        cmap=cmap_precip,
        extend='max',
        transform=ccrs.PlateCarree()
    )


    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )

    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])
 # --- BARRA DE COLOR (HORIZONTAL) ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.05, 0.001, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(mm)", fontsize=11, weight='bold')

    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])

    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)



    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # Frontera principal de México
    mexico_geom = mexico.unary_union
    ax.add_geometries([mexico_geom], crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=5, alpha=0.8, zorder=5,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)])
    ax.add_geometries([mexico_geom], crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='white', linewidth=2, zorder=8,
                      path_effects=[PathEffects.Normal()])

    # Líneas estatales interiores
    try:
        mexico_boundary = mexico_geom.boundary
        all_state_lines = []
        for geometry in mexico.geometry:
            if hasattr(geometry, 'boundary'):
                boundary = geometry.boundary
                if isinstance(boundary, LineString):
                    all_state_lines.append(boundary)
                elif isinstance(boundary, MultiLineString):
                    all_state_lines.extend(boundary.geoms)

        if all_state_lines:
            all_lines = unary_union(all_state_lines)
            interior_lines = all_lines.difference(mexico_boundary.buffer(0.01))
            if not interior_lines.is_empty:
                ax.add_geometries([interior_lines], crs=ccrs.PlateCarree(),
                                  facecolor='none', edgecolor='black', linewidth=0.4,
                                  alpha=0.8, zorder=9)
    except Exception as e:
        print(f"Error dibujando líneas interiores: {e}")

    # --- CIUDADES PRINCIPALES ---
    ciudades_principales = {
        'Mexicali': {'estado': 'Baja California', 'lat': 32.636, 'lon': -115.475},
        'La Paz': {'estado': 'Baja California Sur', 'lat': 24.142, 'lon': -110.313},
        'Hermosillo': {'estado': 'Sonora', 'lat': 29.075, 'lon': -110.958},
        'Culiacán': {'estado': 'Sinaloa', 'lat': 24.79, 'lon': -107.387},
        'Tepic': {'estado': 'Nayarit', 'lat': 21.5, 'lon': -104.9},
        'Guadalajara': {'estado': 'Jalisco', 'lat': 20.666, 'lon': -103.391},
        'Cd de Colima': {'estado': 'Colima', 'lat': 19.1, 'lon': -103.9},
        'Acapulco': {'estado': 'Guerrero', 'lat': 16.862, 'lon': -99.887},
        'Cd de Oaxaca': {'estado': 'Oaxaca', 'lat': 17.06, 'lon': -96.723},
        'Salina Cruz': {'estado': 'Oaxaca', 'lat': 16.2, 'lon': -95.195},
        'Tuxtla': {'estado': 'Chiapas', 'lat': 16.759, 'lon': -93.113},
        'Villahermosa': {'estado': 'Tabasco', 'lat': 17.986, 'lon': -92.93},
        'Cd de Campeche': {'estado': 'Campeche', 'lat': 19.843, 'lon': -90.525},
        'Mérida': {'estado': 'Yucatán', 'lat': 20.975, 'lon': -89.616},
        'Cancún': {'estado': 'Quintana Roo', 'lat': 21.174, 'lon': -86.846},
        'Chetumal': {'estado': 'Quintana Roo', 'lat': 18.514, 'lon': -88.303},
        'Cd de Veracruz': {'estado': 'Veracruz', 'lat': 19.1809, 'lon': -96.142},
        'Poza Rica': {'estado': 'Veracruz', 'lat': 20.533, 'lon': -97.459},
        'Cd de México': {'estado': 'Ciudad de México', 'lat': 19.428, 'lon': -99.127},
        'León': {'estado': 'Guanajuato', 'lat': 21.129, 'lon': -101.673},
        'Cd de Zacatecas': {'estado': 'Zacatecas', 'lat': 22.768, 'lon': -102.581},
        'Cd Mante': {'estado': 'Tamaulipas', 'lat': 22.743, 'lon': -98.973},
        'Reynosa': {'estado': 'Tamaulipas', 'lat': 26.08, 'lon': -98.288},
        'Nuevo Laredo': {'estado': 'Tamaulipas', 'lat': 27.410, 'lon': -99.59},
        'Monterrey': {'estado': 'Nuevo León', 'lat': 25.675, 'lon': -100.318},
        'Piedras Negras': {'estado': 'Coahuila', 'lat': 28.7, 'lon': -100.523},
        'Torreón': {'estado': 'Durango', 'lat': 25.543, 'lon': -103.418},
        'Cd de Durango': {'estado': 'Durango', 'lat': 24.934, 'lon': -104.911},
        'Cd Juárez': {'estado': 'Chihuahua', 'lat': 31.72, 'lon': -106.46},
        'Cd de Chihuahua': {'estado': 'Chihuahua', 'lat': 28.635, 'lon': -106.088}
    }

    # --- Árbol KD para encontrar precipitación más cercana ---
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    # --- DIBUJAR CIUDADES CON ETIQUETAS AZULES ---
    for nombre, datos in ciudades_principales.items():
        lat_ci = datos['lat']
        lon_ci = datos['lon']

        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        # Puntos
        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=4,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        # Nombre ciudad con banner azul
        ax.text(
            lon_ci, lat_ci - 0.18, nombre,
            fontsize=8, color='white', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#1a5490',
                      edgecolor='white', linewidth=0.8, alpha=0.95)
        )

    # Capas adicionales
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                                edgecolor='black', facecolor='none'),
                   linewidth=0.8, alpha=0.3, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.4, edgecolor='black', alpha=0.3, zorder=5)
    ax.add_feature(cfeature.STATES, linewidth=1, edgecolor='black', alpha=0.2, zorder=5)

    # --- LOGO ---
    try:
        logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
        axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower left',
                                bbox_to_anchor=(0.86, 0.04, 1, 1),
                                bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print("Logo no encontrado")

    ax.text(0.97, 0.03, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold', zorder=35,
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.Normal()
            ])

    # --- GUARDAR ---
    nombre_archivo = str(DEFAULT_OUTPUT_DIR / "mexico_mapa_precipitacion.png")
    plt.savefig(nombre_archivo, dpi=800, bbox_inches='tight')
    print(f"Mapa guardado: {nombre_archivo}")
    plt.show()

    return fig, ax


# --- EJECUCIÓN ---
# Asegúrate de tener cargado gdf_estados y tu DataArray de precipitación
# ds_precip debe ser un xarray.DataArray con precipitación en mm
print("📊 Generando mapa de precipitación para México...")
graficar_precipitacion_mexico(ds_multi, gdf_estados=gdf_estados)


#-----NORTEAMÉRICA Y CARIBE - PRECIPITACIÓN MULTIMODELO----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_regional(precip_dataarray, gdf_estados=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para la región completa: Estados Unidos, México,
    Centroamérica, Caribe, Colombia y Venezuela.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    La precipitación se muestra tanto sobre tierra como sobre el mar.

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con todos los estados
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """

    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # Filtrar todos los países de la región
    paises = [
        'United States of America', 'Mexico', 'Guatemala', 'Belize', 'Honduras',
        'El Salvador', 'Nicaragua', 'Costa Rica', 'Panama', 'Colombia', 'Venezuela',
        'Cuba', 'Haiti', 'Dominican Republic', 'Jamaica', 'Puerto Rico',
        'Trinidad and Tobago', 'Bahamas'
    ]
    gdf_regional = gdf_estados[gdf_estados['admin'].isin(paises)].copy()

    # --- PARÁMETROS DE ÁREA (REGIÓN COMPLETA) ---
    lon_min_deg = -125
    lon_max_deg = -58
    lat_min = 0.5
    lat_max = 50

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f" LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f" LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f" Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_max+1, lat_min-1),
            longitude=slice(lon_min_deg, lon_max_deg)
        )
    else:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_min-1, lat_max+1),
            longitude=slice(lon_min_deg, lon_max_deg)
        )

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f" LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f" LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f" Precipitación: Min={precip_vals.min():.2f}, Max={precip_vals.max():.2f}, Mean={precip_vals.mean():.2f} mm")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 600)
    lat_fine = np.linspace(lat.min(), lat.max(), 600)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)

    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA SOLO POR UMBRAL (sin restringir a tierra) ---
    umbral_precip = 5
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_masked_fine)

    if precip_max < 50:
        vmin_colorbar, vmax_colorbar = 1, 50
    elif 50 <= precip_max < 100:
        vmin_colorbar, vmax_colorbar = 0, 100
    elif 100 <= precip_max < 150:
        vmin_colorbar, vmax_colorbar = 0, 200
    else:
        vmin_colorbar, vmax_colorbar = 1, 300

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(22, 14))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)

    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN (SOBRE TIERRA Y MAR) ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )

    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTAS CON SOMBRA 3D ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- FRONTERAS PAÍSES CON EFECTO 3D ---
    for pais in paises:
        gdf_pais = gdf_regional[gdf_regional['admin'] == pais]
        if len(gdf_pais) > 0:
            fronteras = gdf_pais.unary_union.boundary
            ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                              edgecolor='black', facecolor='none', linewidth=2.0, alpha=0.3,
                              path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                              zorder=6)
            ax.add_geometries([fronteras], crs=ccrs.PlateCarree(),
                              edgecolor='white', facecolor='none', linewidth=1.0,
                              path_effects=[PathEffects.Normal()], zorder=7)

    # --- ESTADOS/PROVINCIAS INTERNAS (solo para países principales) ---
    paises_con_estados = ['United States of America', 'Mexico', 'Colombia', 'Venezuela']
    for pais in paises_con_estados:
        estados = gdf_regional[(gdf_regional['admin'] == pais) & (gdf_regional['type'] != 'Country')]
        if len(estados) > 0:
            estados.boundary.plot(ax=ax, edgecolor='black', linewidth=0.25,
                                      zorder=4, transform=ccrs.PlateCarree())

    ## --- BARRA DE COLOR (HORIZONTAL) ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.05, 0.001, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(mm)", fontsize=11, weight='bold')

    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])

    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)

    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES PRINCIPALES DE LA REGIÓN ---
    ciudades_principales = {
        # Estados Unidos (selección)
        #"New York": {'lat': 40.71, 'lon': -74.01},

    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        ax.plot(lon_ci, lat_ci, 'o',
                color='red', markersize=3.5,
                markeredgecolor='white', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=12)

        ax.text(
            lon_ci, lat_ci - 0.25, nombre,
            fontsize=6.5, color='white', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#1a5490',
                      edgecolor='white', linewidth=0.8, alpha=0.95)
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width="7%", height="7%", loc='lower right',
                                    bbox_to_anchor=(-0.9, 0.18, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f"Logo no encontrado en {DEFAULT_LOGO_PATH}")

    # Crédito
    ax.text(0.15, 0.3, 'Creado por MeteOcean',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.4, 0.04,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)

    ax.text(0.21, 0.03, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8.5, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)

    ax.text(0.21, 0.018, 'Creado por ElTiempoconLorenzo',
            transform=ax.transAxes, fontsize=7, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/regional_mapa_precipitacion_multimodelo.png",
                dpi=800, bbox_inches='tight')
    print(f"✅ Mapa guardado: {output_dir}/regional_mapa_precipitacion_multimodelo.png")

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax


# --- USO ---
# Asegúrate de tener cargado gdf_estados y ds_multimodelo de tu clase MultiModeloMeteorologico
fig, ax = graficar_precipitacion_regional(ds_multi, gdf_estados=gdf_estados)


#-----PENÍNSULA IBÉRICA - PRECIPITACIÓN----#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString
from scipy.ndimage import gaussian_filter
from matplotlib.patches import FancyBboxPatch

def graficar_precipitacion_iberia(precip_dataarray, gdf_estados=None):
    """
    Grafica precipitación acumulada para la Península Ibérica (España y Portugal) en mm.

    Parámetros:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray con precipitación en mm con coordenadas 'longitude' y 'latitude'
    gdf_estados : GeoDataFrame
        GeoDataFrame con comunidades autónomas de España y regiones de Portugal
    """

    # --- VALIDACIÓN DE PARÁMETROS ---
    if gdf_estados is None:
        raise ValueError("Debe proporcionar 'gdf_estados'")

    # --- PARÁMETROS DE ÁREA (PENÍNSULA IBÉRICA) ---
    lon_min_deg = -12
    lon_max_deg = 5
    lat_min = 35
    lat_max = 45

    # --- FILTRAR ESPAÑA Y PORTUGAL POR SEPARADO ---
    espana = gdf_estados[gdf_estados['admin'] == 'Spain'].copy()
    portugal = gdf_estados[gdf_estados['admin'] == 'Portugal'].copy()
    iberia = pd.concat([espana, portugal])

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER Y CORREGIR COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    precip_corrected = precip_dataarray.assign_coords(longitude=lon_corrected)

    # --- SELECCIONAR ÁREA (con margen adicional SOLO para los datos) ---
    margen_lon = 2
    margen_lat = 2

    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_max + margen_lat, lat_min - margen_lat),
            longitude=slice(lon_min_deg - margen_lon, lon_max_deg + margen_lon)
        )
    else:
        precip_area = precip_corrected.sel(
            latitude=slice(lat_min - margen_lat, lat_max + margen_lat),
            longitude=slice(lon_min_deg - margen_lon, lon_max_deg + margen_lon)
        )

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    if precip_area.values.ndim == 3:
        precip_vals = precip_area.values[0, :, :]
    else:
        precip_vals = precip_area.values

    # --- VALORES EN MM ---
    precip_vals_mm = precip_vals

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)

    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals_mm.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- APLICAR MÁSCARA SOLO POR UMBRAL ---
    umbral_precip = 3  # mm
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_masked_fine)

    if precip_max < 50:
        vmin_colorbar, vmax_colorbar = 1, 50
    elif 50 <= precip_max < 100:
        vmin_colorbar, vmax_colorbar = 0, 100
    elif 100 <= precip_max < 150:
        vmin_colorbar, vmax_colorbar = 0, 200
    else:
        vmin_colorbar, vmax_colorbar = 1, 300

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- PLOTEO ---

    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')

    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                                   edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)

    # --- COLORES DE PRECIPITACIÓN ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels,
        cmap=cmap_precip,
        extend='max',
        transform=ccrs.PlateCarree()
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )

    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- BARRA DE COLOR (VERTICAL) ---
    cax = inset_axes(ax, width="3%", height="40%", loc='lower left',
                 bbox_to_anchor=(0.02, 0.15, 1, 1),
                 bbox_transform=ax.transAxes, borderpad=0)

    cbar = plt.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label("(L/m²)", fontsize=11, weight='bold')

    label_obj = cbar.ax.yaxis.get_label()
    label_obj.set_path_effects([
    PathEffects.withStroke(linewidth=3, foreground='white'),
    PathEffects.Normal()
        ])

    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)

    for label in cbar.ax.get_yticklabels():
      label.set_fontweight('bold')
      label.set_path_effects([
      PathEffects.withStroke(linewidth=3, foreground='white'),
      PathEffects.Normal()
      ])


    # --- FRONTERAS: ESPAÑA Y PORTUGAL POR SEPARADO ---
    # España
    espana_geom = espana.unary_union
    ax.add_geometries([espana_geom], crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=5, alpha=0.8, zorder=5,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)])
    ax.add_geometries([espana_geom], crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='white', linewidth=2, zorder=8,
                      path_effects=[PathEffects.Normal()])

    # Portugal
    portugal_geom = portugal.unary_union
    ax.add_geometries([portugal_geom], crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=5, alpha=0.8, zorder=5,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)])
    ax.add_geometries([portugal_geom], crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='white', linewidth=2, zorder=8,
                      path_effects=[PathEffects.Normal()])

    # --- LÍNEAS INTERIORES: COMUNIDADES AUTÓNOMAS (ESPAÑA) Y REGIONES (PORTUGAL) ---
    try:
        # Líneas de comunidades autónomas de España
        espana_boundary = espana_geom.boundary
        all_comunidades_lines = []
        for geometry in espana.geometry:
            if hasattr(geometry, 'boundary'):
                boundary = geometry.boundary
                if isinstance(boundary, LineString):
                    all_comunidades_lines.append(boundary)
                elif isinstance(boundary, MultiLineString):
                    all_comunidades_lines.extend(boundary.geoms)

        if all_comunidades_lines:
            all_lines_esp = unary_union(all_comunidades_lines)
            interior_lines_esp = all_lines_esp.difference(espana_boundary.buffer(0.01))
            if not interior_lines_esp.is_empty:
                ax.add_geometries([interior_lines_esp], crs=ccrs.PlateCarree(),
                                  facecolor='none', edgecolor='black', linewidth=0.4,
                                  alpha=0.8, zorder=9)

        # Líneas de regiones de Portugal
        portugal_boundary = portugal_geom.boundary
        all_portugal_lines = []
        for geometry in portugal.geometry:
            if hasattr(geometry, 'boundary'):
                boundary = geometry.boundary
                if isinstance(boundary, LineString):
                    all_portugal_lines.append(boundary)
                elif isinstance(boundary, MultiLineString):
                    all_portugal_lines.extend(boundary.geoms)

        if all_portugal_lines:
            all_lines_pt = unary_union(all_portugal_lines)
            interior_lines_pt = all_lines_pt.difference(portugal_boundary.buffer(0.01))
            if not interior_lines_pt.is_empty:
                ax.add_geometries([interior_lines_pt], crs=ccrs.PlateCarree(),
                                  facecolor='none', edgecolor='black', linewidth=0.4,
                                  alpha=0.8, zorder=9)

    except Exception as e:
        print(f"Error dibujando líneas interiores: {e}")

    # --- CAPITALES DE COMUNIDADES AUTÓNOMAS ---
    capitales_autonomas = {
        # ESPAÑA
        'Ceuta': {'lat': 35.8894, 'lon': -5.3213},
        'Melilla': {'lat': 35.2943, 'lon': -2.9526},
        'Álava': {'lat': 42.85, 'lon': -2.69},  # País Vasco
        'Albacete': {'lat': 38.99, 'lon': -1.86},  # Castilla-La Mancha
        'Alicante': {'lat': 38.35, 'lon': -0.48},  # Comunidad Valenciana
        'Almería': {'lat': 36.84, 'lon': -2.46},  # Andalucía
        'Asturias': {'lat': 43.37, 'lon': -5.86},  # Asturias
        'Ávila': {'lat': 40.65, 'lon': -4.70},  # Castilla y León
        'Badajoz': {'lat': 38.88, 'lon': -6.97},  # Extremadura
        'Barcelona': {'lat': 41.39, 'lon': 2.17},  # Cataluña
        'Burgos': {'lat': 42.34, 'lon': -3.70},  # Castilla y León
        'Cáceres': {'lat': 39.48, 'lon': -6.37},  # Extremadura
        'Cádiz': {'lat': 36.53, 'lon': -6.29},  # Andalucía
        'Cantabria': {'lat': 43.46, 'lon': -3.81},  # Cantabria
        'Castellón': {'lat': 39.98, 'lon': -0.04},  # Comunidad Valenciana
        'Ciudad Real': {'lat': 38.99, 'lon': -3.93},  # Castilla-La Mancha
        'Córdoba': {'lat': 37.88, 'lon': -4.78},  # Andalucía
        'Cuenca': {'lat': 40.07, 'lon': -2.13},  # Castilla-La Mancha
        'Girona': {'lat': 41.98, 'lon': 2.82},  # Cataluña
        'Granada': {'lat': 37.18, 'lon': -3.60},  # Andalucía
        'Guadalajara': {'lat': 40.63, 'lon': -3.17},  # Castilla-La Mancha
        'Guipúzkoa': {'lat': 43.31, 'lon': -1.98},  # País Vasco
        'Huelva': {'lat': 37.26, 'lon': -6.95},  # Andalucía
        'Huesca': {'lat': 42.14, 'lon': -0.41},  # Aragón
        'Ibiza': {'lat': 38.9105, 'lon': 1.4247},  # Illes Balears
        'Mallorca': {'lat': 39.61, 'lon': 2.97},  # Illes Balears
        'Menorca': {'lat': 40.0002, 'lon': 3.84},  # Illes Balears
        'Jaén': {'lat': 37.77, 'lon': -3.79},  # Andalucía
        'La Coruña': {'lat': 43.36, 'lon': -8.41},  # Galicia
        'La Rioja': {'lat': 42.29, 'lon': -2.54},  # La Rioja
        'León': {'lat': 42.60, 'lon': -5.57},  # Castilla y León
        'Lérida': {'lat': 41.61, 'lon': 0.62},  # Cataluña
        'Lugo': {'lat': 43.01, 'lon': -7.56},  # Galicia
        'Madrid': {'lat': 40.42, 'lon': -3.70},  # Comunidad de Madrid
        'Málaga': {'lat': 36.72, 'lon': -4.42},  # Andalucía
        'Murcia': {'lat': 37.99, 'lon': -1.13},  # Región de Murcia
        'Navarra': {'lat': 42.69, 'lon': -1.68},  # Navarra
        'Orense': {'lat': 42.34, 'lon': -7.86},  # Galicia
        'Palencia': {'lat': 42.01, 'lon': -4.53},  # Castilla y León
        'Pontevedra': {'lat': 42.43, 'lon': -8.64},  # Galicia
        'Salamanca': {'lat': 40.97, 'lon': -5.66},  # Castilla y León
        'Segovia': {'lat': 40.95, 'lon': -4.12},  # Castilla y León
        'Sevilla': {'lat': 37.39, 'lon': -5.99},  # Andalucía
        'Soria': {'lat': 41.77, 'lon': -2.47},  # Castilla y León
        'Tarragona': {'lat': 41.12, 'lon': 1.25},  # Cataluña
        'Teruel': {'lat': 40.34, 'lon': -1.11},  # Aragón
        'Toledo': {'lat': 39.86, 'lon': -4.03},  # Castilla-La Mancha
        'Valencia': {'lat': 39.47, 'lon': -0.38},  # Comunidad Valenciana
        'Valladolid': {'lat': 41.65, 'lon': -4.72},  # Castilla y León
        'Vizkaya': {'lat': 43.26, 'lon': -2.93},  # País Vasco
        'Zamora': {'lat': 41.50, 'lon': -5.74},  # Castilla y León
        'Zaragoza': {'lat': 41.65, 'lon': -0.89},  # Aragón



        # PORTUGAL
        'Lisboa': {'lat': 38.722, 'lon': -9.139},  # Lisboa
        'Porto': {'lat': 41.158, 'lon': -8.629},  # Norte
        'Coimbra': {'lat': 40.211, 'lon': -8.429},  # Centro
        'Évora': {'lat': 38.571, 'lon': -7.907},  # Alentejo
        'Faro': {'lat': 37.017, 'lon': -7.930},  # Algarve
    }

    # --- Árbol KD para encontrar precipitación más cercana ---
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    # --- DIBUJAR CAPITALES CON ETIQUETAS AZULES ---
    for nombre, datos in capitales_autonomas.items():
        lat_ci = datos['lat']
        lon_ci = datos['lon']

        # Verificar si está dentro del área visible
        if not (lon_min_deg <= lon_ci <= lon_max_deg and lat_min <= lat_ci <= lat_max):
            continue

        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        # Puntos
        #ax.plot(lon_ci, lat_ci, 'o',
                #color='red', markersize=4,
                #markeredgecolor='white', markeredgewidth=1,
                #transform=ccrs.PlateCarree(), zorder=12)

        # Nombre ciudad con banner azul
        ax.text(
            lon_ci, lat_ci, nombre,
            fontsize=9, color='white', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#1a5490',
                      edgecolor='white', linewidth=0.8, alpha=0.95)
        )

    # --- CAPAS ADICIONALES ---
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                                edgecolor='black', facecolor='none'),
                   linewidth=0.8, alpha=0.3, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.4, edgecolor='black', alpha=0.3, zorder=5)
    ax.add_feature(cfeature.STATES, linewidth=1, edgecolor='black', alpha=0.2, zorder=5)

    # --- LOGO (OPCIONAL) ---
    try:
        logo_img = mpimg.imread(DEFAULT_SPAIN_LOGO_PATH)
        axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower left',
                                bbox_to_anchor=(0.86, 0.04, 1, 1),
                                bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print("Logo no encontrado (opcional)")

    ax.text(0.97, 0.03, 'Creado por MeteoEspaña',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold', zorder=35,
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.4, 0.04,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)
    ax.text(0.21, 0.03, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8.5, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)

    ax.text(0.21, 0.015, 'Creado por MeteoEspaña',
            transform=ax.transAxes, fontsize=7, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    nombre_archivo = str(DEFAULT_OUTPUT_DIR / "iberia_mapa_precipitacion.png")
    plt.savefig(nombre_archivo, dpi=800, bbox_inches='tight')
    print(f"Mapa guardado: {nombre_archivo}")
    plt.show()

    return fig, ax


# --- EJECUCIÓN ---
# Asegúrate de tener cargado gdf_estados y tu DataArray de precipitación
# ds_precip debe ser un xarray.DataArray con precipitación en mm
print("📊 Generando mapa de precipitación para la Península Ibérica...")
graficar_precipitacion_iberia(ds_multi, gdf_estados=gdf_estados)


#-----ISLAS CANARIAS - PRECIPITACIÓN MULTIMODELO - REJILLA COMÚN ----#
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
import xarray as xr
from matplotlib.patches import FancyBboxPatch
from IPython.display import display

def graficar_precipitacion_canarias(precip_dataarray, canarias=None, gdf_estados=None, mostrar_logo=True):
    """
    Grafica precipitación acumulada para Islas Canarias y guarda la figura en outputs/maps.
    Ajusta dinámicamente el rango de colores según el valor máximo de precipitación.
    """
    # --- PARÁMETROS DE ÁREA ---
    lon_min_deg, lon_max_deg = -19.5, -12
    lat_min, lat_max = 26.5, 30.5

    # --- VERIFICAR COORDENADAS ---
    if 'longitude' not in precip_dataarray.coords or 'latitude' not in precip_dataarray.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")

    # --- EXTRAER COORDENADAS ---
    lon_orig = precip_dataarray.longitude.values
    lat_orig = precip_dataarray.latitude.values

    print(f"Coordenadas originales:")
    print(f"  LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} puntos")
    print(f"  LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} puntos")
    print(f"  Orden LAT: {'Descendente (N→S)' if lat_orig[0] > lat_orig[-1] else 'Ascendente (S→N)'}")

    precip_corrected = precip_dataarray

    # --- SELECCIONAR ÁREA ---
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max, lat_min),
                                           longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max),
                                           longitude=slice(lon_min_deg, lon_max_deg))

    lat = precip_area.latitude.values
    lon = precip_area.longitude.values

    print(f"\nÁrea seleccionada:")
    print(f"  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} puntos")
    print(f"  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} puntos")

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f"El área seleccionada está vacía. Verifica los límites: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]")

    precip_vals = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f"  Precipitación: Min={precip_vals.min():.2f}, Max={precip_vals.max():.2f}, Mean={precip_vals.mean():.2f} mm")

    # --- MALLA ORIGINAL ---
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- INTERPOLACIÓN A REJILLA MÁS FINA ---
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')

    # --- SUAVIZADO GAUSSIANO ---
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)

    # --- MÁSCARA ---
    umbral_precip = 1
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)

    # --- RANGO DE COLOR DINÁMICO ---
    precip_max = np.nanmax(precip_fine_smooth)
    print(precip_max)
    if precip_max < 50:
        vmin_colorbar, vmax_colorbar = 1, 50
    elif 50 <= precip_max < 100:
        vmin_colorbar, vmax_colorbar = 0, 100
    else:
        vmin_colorbar, vmax_colorbar = 0, 150

    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)

    # --- CREAR FIGURA ---
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- TERRITORIO ---
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)

    if canarias is None and gdf_estados is not None:
        canarias = gdf_estados[gdf_estados['admin'] == 'Spain'].copy()
    elif canarias is None:
        raise ValueError("Debe proporcionar 'canarias' o 'gdf_estados'")

    canarias.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=5)

    # --- COLORES ---
    colors_precip = [
        '#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300',
        '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitacion', colors_precip)

    # --- CONTORNO DE PRECIPITACIÓN ---
    cf = ax.contourf(
        lon2d_fine, lat2d_fine, precip_masked_fine,
        levels=levels, cmap=cmap_precip, extend='max',
        transform=ccrs.PlateCarree(), zorder=3
    )

    # --- ISOLÍNEAS ---
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(
        lon2d_fine, lat2d_fine, precip_fine_smooth,
        levels=contour_levels, colors='black', linewidths=0.6,
        alpha=0.7, transform=ccrs.PlateCarree(), zorder=4
    )
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground='white'),
            PathEffects.Normal()
        ])

    # --- COSTA CON SOMBRA ---
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3,
                      path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)],
                      zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=1.3,
                      path_effects=[PathEffects.Normal()], zorder=7)

    # --- BARRA DE COLOR ---
    cax = inset_axes(ax, width="42%", height="5%", loc='lower left',
                     bbox_to_anchor=(-0.015, 0.0, 0.9, 0.9),
                     bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label("(L/m²)", fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white'),
        PathEffects.Normal()
    ])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white'),
            PathEffects.Normal()
        ])

    # --- CIUDADES ---
    ciudades_principales = {
        # Tenerife
        "Tenerife": {'lat': 28.3191, 'lon': -16.27},

        # Gran Canaria
        "Gran Canaria": {'lat': 27.7605, 'lon': -15.5838},

        # Lanzarote
        "Lanzarote": {'lat': 28.9394, 'lon': -13.5316},

        # Fuerteventura
        "Fuerteventura": {'lat': 28.06, 'lon': -13.9113},

        # La Palma
        "La Palma": {'lat': 28.4647, 'lon': -17.8391},

        # La Gomera
        "La Gomera": {'lat': 28.0263, 'lon': -17.2322},

        # El Hierro
        "El Hierro": {'lat': 27.6433, 'lon': -17.9783},
    }

    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()

    for nombre, datos in ciudades_principales.items():
        lat_ci, lon_ci = datos['lat'], datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        precip = precip_values[idx]

        #ax.plot(lon_ci, lat_ci, 'o',
                #color='red', markersize=4,
                #markeredgecolor='white', markeredgewidth=1,
                #transform=ccrs.PlateCarree(), zorder=12)

        ax.text(
            lon_ci, lat_ci - 0.08, nombre,
            fontsize=7.4, color='#2C3E50', weight='bold',
            ha='center', va='top', zorder=15,
            transform=ccrs.PlateCarree(),
            path_effects=[
                PathEffects.withStroke(linewidth=2, foreground='white'),
                PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                PathEffects.Normal()
            ]
        )

    # --- LOGO ---
    if mostrar_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_SPAIN_LOGO_PATH)
            axins_logo = inset_axes(ax, width="9.5%", height="9.5%", loc='lower right',
                                    bbox_to_anchor=(-0.05, 0.03, 1, 1),
                                    bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print("Logo no encontrado")

    # Crédito
    ax.text(1.0, 0.15, 'Creado por MeteoEspaña',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='black', fontstyle='italic', fontweight='bold',
            path_effects=[
                PathEffects.withStroke(linewidth=2.5, foreground='white'),
                PathEffects.Normal()
            ])

    # --- BANNER ---
    banner_box = FancyBboxPatch(
        (0.01, 0.01), 0.48, 0.055,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.95,
        zorder=20
    )
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Multimodelo Experimental Híbrido (Física + IA)',
            transform=ax.transAxes, fontsize=8, ha='center', va='center',
            color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Creado por MeteoEspaña',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='center',
            color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)

    # --- GUARDAR ---
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/canarias_mapa_precipitacion_multimodelo.png", dpi=800, bbox_inches='tight')

    # --- MOSTRAR AUTOMÁTICAMENTE EN COLAB ---
    plt.show()

    return fig, ax

# --- USO ---
canarias = gdf_estados[gdf_estados['admin'] == 'Spain'].copy()
fig, ax = graficar_precipitacion_dataarray(ds_multi, canarias=canarias)
