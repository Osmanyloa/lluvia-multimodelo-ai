"""Regional accumulated precipitation map renderers.

The functions in this module expect an ``xarray.DataArray`` with ``longitude``
and ``latitude`` coordinates. Some regions also require Natural Earth or Census
boundary layers supplied as ``geopandas.GeoDataFrame`` objects.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

DEFAULT_OUTPUT_DIR = Path("outputs/maps")
DEFAULT_LOGO_PATH = Path("assets/logo.png")
DEFAULT_SPAIN_LOGO_PATH = Path("assets/logo_spain.png")

def plot_cuba_precipitation(precipitation, cuba_boundary=None, admin_boundaries=None, show_logo=True):
    """
    Plot accumulated precipitation for Cuba and save the figure to outputs/maps.
    Dynamically adjusts the color range from the maximum precipitation value.
    """
    lon_min_deg, lon_max_deg = (-87, -73)
    lat_min, lat_max = (18, 25)
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f'  LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f'  LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f"  Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f'  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f'  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f'  Precipitation: Min={precip_values.min():.2f}, Max={precip_values.max():.2f}, Mean={precip_values.mean():.2f} mm')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 500)
    lat_fine = np.linspace(lat.min(), lat.max(), 500)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 3
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_fine_smooth)
    print(precipitation_max)
    if precipitation_max < 50:
        vmin_colorbar, vmax_colorbar = (1, 50)
    elif 50 <= precipitation_max < 100:
        vmin_colorbar, vmax_colorbar = (0, 100)
    else:
        vmin_colorbar, vmax_colorbar = (0, 150)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    if cuba_boundary is None and admin_boundaries is not None:
        cuba_boundary = admin_boundaries[admin_boundaries['admin'] == 'Cuba'].copy()
    elif cuba_boundary is None:
        raise ValueError("You must provide either cuba_boundary or admin_boundaries.")
    cuba_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=5)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.088, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(mm)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'PR': {'lat': 22.42, 'lon': -83.7}, 'HAB': {'lat': 23.17, 'lon': -82.28}, 'MTZ': {'lat': 23.04, 'lon': -81.58}, 'Nueva Gerona': {'lat': 21.88, 'lon': -82.8}, 'Colón': {'lat': 22.72, 'lon': -80.91}, 'SC': {'lat': 22.41, 'lon': -79.97}, 'SSP': {'lat': 21.93, 'lon': -79.43}, 'CAV': {'lat': 21.84, 'lon': -78.76}, 'CMG': {'lat': 21.38, 'lon': -77.91}, 'Moa': {'lat': 20.66, 'lon': -74.945}, 'HOL': {'lat': 20.89, 'lon': -76.26}, 'Bayamo': {'lat': 20.38, 'lon': -76.64}, 'SCU': {'lat': 20.02, 'lon': -75.82}, 'Maisí': {'lat': 20.24, 'lon': -74.15}, 'LTU': {'lat': 20.96, 'lon': -76.95}, 'CFG': {'lat': 22.15, 'lon': -80.44}, 'ART': {'lat': 22.81, 'lon': -82.76}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.08, name, fontsize=9, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.05, 0.03, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print('Logo not found')
    ax.text(0.5, 0.39, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteOcean', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/cuba_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    plt.show()
    return (fig, ax)

def plot_central_america_precipitation(precipitation, admin_boundaries=None, show_logo=True):
    """
    Plot accumulated precipitation for Central America and save the figure.
    Dynamically adjusts the color range from the maximum precipitation value.

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with administrative boundaries
    mostrar_logo : bool
        Whether to show the logo (default: True)
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    countries = ['Guatemala', 'Belize', 'Honduras', 'El Salvador', 'Nicaragua', 'Costa Rica', 'Panama']
    central_america_boundaries = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg, lon_max_deg = (-94, -74)
    lat_min, lat_max = (7, 18.7)
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f' LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f' LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f" Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max + 1, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max + 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f' LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f' LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f' Precipitation: Min={precip_values.min():.2f}, Max={precip_values.max():.2f}, Mean={precip_values.mean():.2f} mm')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 500)
    lat_fine = np.linspace(lat.min(), lat.max(), 500)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 3
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    if precipitation_max < 50:
        vmin_colorbar, vmax_colorbar = (1, 50)
    elif 50 <= precipitation_max < 100:
        vmin_colorbar, vmax_colorbar = (0, 100)
    elif 100 <= precipitation_max < 150:
        vmin_colorbar, vmax_colorbar = (0, 150)
    else:
        vmin_colorbar, vmax_colorbar = (1, 250)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        country_boundaries = central_america_boundaries[central_america_boundaries['admin'] == country]
        if len(country_boundaries) > 0:
            borders = country_boundaries.unary_union.boundary
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        provinces = central_america_boundaries[(central_america_boundaries['admin'] == country) & (central_america_boundaries['type'] != 'Country')]
        if len(provinces) > 0:
            provinces.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=4, transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.07, 0.088, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(mm)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Guatemala': {'lat': 14.63, 'lon': -90.53}, 'Quetzaltenango': {'lat': 14.83, 'lon': -91.52}, 'Cobán': {'lat': 15.47, 'lon': -90.37}, 'Flores': {'lat': 16.91, 'lon': -89.89}, 'Tegucigalpa': {'lat': 14.1, 'lon': -87.22}, 'San Pedro Sula': {'lat': 15.5, 'lon': -88.03}, 'La Ceiba': {'lat': 15.75, 'lon': -86.79}, 'Choluteca': {'lat': 13.3, 'lon': -87.2}, 'Tocoa': {'lat': 15.65, 'lon': -85.99}, 'San Salvador': {'lat': 13.69, 'lon': -89.2}, 'San Miguel': {'lat': 13.48, 'lon': -88.18}, 'Managua': {'lat': 12.13, 'lon': -86.25}, 'León': {'lat': 12.44, 'lon': -86.88}, 'Estelí': {'lat': 13.09, 'lon': -86.36}, 'Rivas': {'lat': 11.43, 'lon': -85.82}, 'Bilwi': {'lat': 14.03, 'lon': -83.39}, 'San Andrés': {'lat': 12.583, 'lon': -81.7}, 'Providencia': {'lat': 13.3733, 'lon': -81.3627}, 'San José': {'lat': 9.93, 'lon': -84.08}, 'Liberia': {'lat': 10.63, 'lon': -85.44}, 'Ciudad de Panamá': {'lat': 9.01, 'lon': -79.52}, 'David': {'lat': 8.43, 'lon': -82.43}, 'Santiago de Veraguas': {'lat': 8.1, 'lon': -80.97}, 'Belmopán': {'lat': 17.25, 'lon': -88.77}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.08, name, fontsize=7.4, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.198, 0.25, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f'Logo not found en {DEFAULT_LOGO_PATH}')
    ax.text(0.8, 0.4, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteOcean', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/central_america_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/central_america_multimodel_precipitation_map.png')
    plt.show()
    return (fig, ax)

def plot_hispaniola_precipitation(precipitation, admin_boundaries=None, show_logo=True):
    """
    Plot accumulated precipitation for Hispaniola (Haiti and the Dominican Republic).
    Dynamically adjusts the color range from the maximum precipitation value.
    Precipitation is shown over land and ocean.

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with administrative boundaries
    mostrar_logo : bool
        Whether to show the logo (default: True)
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    haiti = admin_boundaries[admin_boundaries['admin'] == 'Haiti'].copy()
    republica_dominicana = admin_boundaries[admin_boundaries['admin'] == 'Dominican Republic'].copy()
    hispaniola = pd.concat([haiti, republica_dominicana])
    lon_min_deg = -76
    lon_max_deg = -67
    lat_min = 16
    lat_max = 20.8
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f' LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f' LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f" Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max + 1, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max + 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f' LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f' LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f' Precipitation: Min={precip_values.min():.2f}, Max={precip_values.max():.2f}, Mean={precip_values.mean():.2f} mm')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 3
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    if precipitation_max < 50:
        vmin_colorbar, vmax_colorbar = (1, 50)
    elif 50 <= precipitation_max < 100:
        vmin_colorbar, vmax_colorbar = (0, 100)
    elif 100 <= precipitation_max < 150:
        vmin_colorbar, vmax_colorbar = (0, 150)
    else:
        vmin_colorbar, vmax_colorbar = (1, 200)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    frontera_internacional = haiti.unary_union.intersection(republica_dominicana.unary_union)
    ax.add_geometries([frontera_internacional], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries([frontera_internacional], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    provincias_haiti = admin_boundaries[(admin_boundaries['admin'] == 'Haiti') & (admin_boundaries['type'] != 'Country')]
    provincias_haiti.boundary.plot(ax=ax, edgecolor='black', linewidth=0.3, zorder=4, transform=ccrs.PlateCarree())
    provincias_rd = admin_boundaries[(admin_boundaries['admin'] == 'Dominican Republic') & (admin_boundaries['type'] != 'Country')]
    provincias_rd.boundary.plot(ax=ax, edgecolor='black', linewidth=0.3, zorder=4, transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.088, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(mm)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Puerto Príncipe': {'lat': 18.54, 'lon': -72.34}, 'Santiago': {'lat': 19.45, 'lon': -70.7}, 'Santo Domingo': {'lat': 18.48, 'lon': -69.9}, 'Cotuí': {'lat': 19.08, 'lon': -70.16}, 'San Juan': {'lat': 18.8, 'lon': -71.22}, 'Barahona': {'lat': 18.2, 'lon': -71.1}, 'Punta Cana': {'lat': 18.58, 'lon': -68.4}, 'Jérémie': {'lat': 18.64, 'lon': -74.12}, 'La Romana': {'lat': 18.43, 'lon': -68.97}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.08, name, fontsize=7.4, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.198, 0.19, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f'Logo not found en {DEFAULT_LOGO_PATH}')
    ax.text(0.8, 0.35, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteOcean', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/hispaniola_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/hispaniola_multimodel_precipitation_map.png')
    plt.show()
    return (fig, ax)

def plot_lesser_antilles_precipitation(precipitation, admin_boundaries=None, show_logo=True):
    """
    Plot accumulated precipitation for the Lesser Antilles and save the figure to outputs/maps.
    Dynamically adjusts the color range from the maximum precipitation value.
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    countries = ['Antigua and Barbuda', 'Dominica', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Grenada', 'Barbados', 'Trinidad and Tobago', 'Saint Kitts and Nevis', 'Martinique', 'Guadeloupe', 'Montserrat', 'Anguilla', 'British Virgin Islands', 'United States Virgin Islands', 'Aruba', 'Curaçao', 'Sint Maarten', 'Saint Barthélemy', 'Bonaire', 'France']
    antillas_menores = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg = -71
    lon_max_deg = -57
    lat_min = 9.5
    lat_max = 19
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f'  LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f'  LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f"  Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f'  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f'  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f'  Precipitation: Min={precip_values.min():.2f}, Max={precip_values.max():.2f}, Mean={precip_values.mean():.2f} mm')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 3
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_fine_smooth)
    print(f'Precipitation maximum: {precipitation_max:.2f} mm')
    if precipitation_max < 50:
        vmin_colorbar, vmax_colorbar = (1, 50)
    elif 50 <= precipitation_max < 100:
        vmin_colorbar, vmax_colorbar = (0, 100)
    else:
        vmin_colorbar, vmax_colorbar = (0, 150)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.35, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(mm)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Islas Vírgenes Británicas': {'lat': 18.43, 'lon': -64.62}, 'Anguila': {'lat': 18.22, 'lon': -63.05}, 'San Cristóbal': {'lat': 17.3, 'lon': -62.72}, 'Antigua': {'lat': 17.12, 'lon': -61.85}, 'Guadalupe': {'lat': 16.24, 'lon': -61.53}, 'Dominica': {'lat': 15.3, 'lon': -61.39}, 'Martinica': {'lat': 14.61, 'lon': -61.08}, 'Santa Lucía': {'lat': 14.01, 'lon': -60.99}, 'San Vicente': {'lat': 13.1633, 'lon': -61.2233}, 'Barbados': {'lat': 13.1, 'lon': -59.61}, 'Granada': {'lat': 12.05, 'lon': -61.75}, 'Puerto España': {'lat': 10.6702, 'lon': -61.5038}, 'Tobago': {'lat': 11.18, 'lon': -60.74}, 'Aruba': {'lat': 12.52, 'lon': -70.03}, 'Curazao': {'lat': 12.11, 'lon': -68.93}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.08, name, fontsize=7.4, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.55, 0.56, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print('Logo not found')
    ax.text(0.5, 0.7, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteOcean', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/lesser_antilles_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    plt.show()
    return (fig, ax)

def plot_puerto_rico_precipitation(precipitation, admin_boundaries=None, puerto_rico_counties=None, show_logo=True):
    """
    Plot accumulated precipitation for Puerto Rico.
    Dynamically adjusts the color range from the maximum precipitation value.
    Precipitation is shown over land and ocean.
    (Conversion de mm to inches)

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with administrative boundaries
    municipios_pr : GeoDataFrame
        GeoDataFrame con municipios de Puerto Rico (opcional)
    mostrar_logo : bool
        Whether to show the logo (default: True)
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    puerto_rico = admin_boundaries[admin_boundaries['admin'] == 'Puerto Rico'].copy()
    lon_min_deg = -68.0
    lon_max_deg = -65
    lat_min = 17.25
    lat_max = 19.0
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f' LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f' LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f" Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max + 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f' LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f' LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    precip_values = precip_values / 25.4
    print(f' Precipitation (en inches): Min={precip_values.min():.2f}, Max={precip_values.max():.2f}, Mean={precip_values.mean():.2f}')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 0.1
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    if precipitation_max < 2:
        vmin_colorbar, vmax_colorbar = (0, 2)
    elif 2 <= precipitation_max < 4:
        vmin_colorbar, vmax_colorbar = (0, 4)
    else:
        vmin_colorbar, vmax_colorbar = (0, 6)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5), 2))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%.1f', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    puerto_rico.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=5, transform=ccrs.PlateCarree())
    if puerto_rico_counties is not None:
        puerto_rico_counties.boundary.plot(ax=ax, edgecolor='grey', linewidth=0.2, alpha=0.8, zorder=8, transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.088, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal', ticks=np.arange(vmin_colorbar, vmax_colorbar + 0.5, 0.5))
    cbar.set_label('(inches)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'San Juan': {'lat': 18.46, 'lon': -66.105}, 'Ponce': {'lat': 18.01, 'lon': -66.61}, 'Caguas': {'lat': 18.24, 'lon': -66.04}, 'Mayagüez': {'lat': 18.2, 'lon': -67.14}, 'Arecibo': {'lat': 18.47, 'lon': -66.72}, 'Fajardo': {'lat': 18.33, 'lon': -65.65}, 'Aguadilla': {'lat': 18.43, 'lon': -67.15}, 'Humacao': {'lat': 18.15, 'lon': -65.83}, 'Utuado': {'lat': 18.27, 'lon': -66.7}, 'Guayama': {'lat': 17.98, 'lon': -66.11}, 'Vieques': {'lat': 18.12, 'lon': -65.44}, 'Culebra': {'lat': 18.32, 'lon': -65.29}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.02, name, fontsize=7.4, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.198, 0.2, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f'Logo not found en {DEFAULT_LOGO_PATH}')
    ax.text(0.5, 0.34, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteOcean', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/puerto_rico_multimodel_precipitation_map_inches.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/puerto_rico_multimodel_precipitation_map_inches.png')
    plt.show()
    return (fig, ax)

def plot_colombia_venezuela_precipitation(precipitation, admin_boundaries=None, show_logo=True):
    """
    Plot accumulated precipitation for Colombia and Venezuela.
    Dynamically adjusts the color range from the maximum precipitation value.
    Precipitation is shown over land and ocean.

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with administrative boundaries
    mostrar_logo : bool
        Whether to show the logo (default: True)
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    countries = ['Colombia', 'Venezuela']
    gdf_cv = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg = -80
    lon_max_deg = -58
    lat_min = 0.5
    lat_max = 13
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f' LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f' LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f" Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max + 1, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max + 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f' LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f' LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f' Precipitation: Min={precip_values.min():.2f}, Max={precip_values.max():.2f}, Mean={precip_values.mean():.2f} mm')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 5
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    if precipitation_max < 50:
        vmin_colorbar, vmax_colorbar = (1, 50)
    elif 50 <= precipitation_max < 100:
        vmin_colorbar, vmax_colorbar = (0, 100)
    elif 100 <= precipitation_max < 150:
        vmin_colorbar, vmax_colorbar = (0, 200)
    else:
        vmin_colorbar, vmax_colorbar = (1, 300)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        country_boundaries = gdf_cv[gdf_cv['admin'] == country]
        if len(country_boundaries) > 0:
            borders = country_boundaries.unary_union.boundary
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        provinces = gdf_cv[(gdf_cv['admin'] == country) & (gdf_cv['type'] != 'Country')]
        if len(provinces) > 0:
            provinces.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=4, transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='2.5%', height='45%', loc='lower left', bbox_to_anchor=(0.02, 0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label('(mm)', fontsize=10, weight='bold')
    label_obj = cbar.ax.yaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Bogotá': {'lat': 4.60971, 'lon': -74.08175}, 'Medellín': {'lat': 6.2442, 'lon': -75.5812}, 'Cali': {'lat': 3.4516, 'lon': -76.532}, 'Barranquilla': {'lat': 10.9685, 'lon': -74.7813}, 'Montería': {'lat': 8.7471, 'lon': -75.8894}, 'Pereira': {'lat': 4.8143, 'lon': -75.6946}, 'Cúcuta': {'lat': 7.89, 'lon': -72.4963}, 'Cartagena': {'lat': 10.395, 'lon': -75.4833}, 'Valledupar': {'lat': 10.4719, 'lon': -73.2527}, 'Maicao': {'lat': 11.3775, 'lon': -72.2383}, 'Bucaramanga': {'lat': 7.1238, 'lon': -73.1216}, 'Caracas': {'lat': 10.488, 'lon': -66.8792}, 'Maracaibo': {'lat': 10.6539, 'lon': -71.64597}, 'Valencia': {'lat': 10.162, 'lon': -68.0077}, 'Maturín': {'lat': 9.75, 'lon': -63.18}, 'Barcelona': {'lat': 10.1363, 'lon': -64.6862}, 'Ciudad Bolívar': {'lat': 8.0833, 'lon': -63.6}, 'Mérida': {'lat': 8.57, 'lon': -71.18}, 'Barquisimeto': {'lat': 10.0683, 'lon': -69.3452}, 'Coro': {'lat': 11.398, 'lon': -69.6794}, 'Cd Guayana': {'lat': 8.3525, 'lon': -62.643}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.08, name, fontsize=7.4, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.12, 0.01, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f'Logo not found en {DEFAULT_LOGO_PATH}')
    ax.text(0.9, 0.15, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteOcean', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/colombia_venezuela_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/colombia_venezuela_multimodel_precipitation_map.png')
    plt.show()
    return (fig, ax)

def plot_united_states_precipitation(precipitation, admin_boundaries=None, show_logo=True):
    """
    Plot accumulated precipitation for the United States.
    Dynamically adjusts the color range from the maximum precipitation value.
    Precipitation is shown over land and ocean.

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with administrative boundaries
    mostrar_logo : bool
        Whether to show the logo (default: True)
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    countries = ['United States of America']
    gdf_usa = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg = -125
    lon_max_deg = -66
    lat_min = 24
    lat_max = 50
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f' LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f' LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f" Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max + 1, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max + 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f' LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f' LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    precip_values_inches = precip_values / 25.4
    print(f' Precipitation: Min={precip_values_inches.min():.2f}, Max={precip_values_inches.max():.2f}, Mean={precip_values_inches.mean():.2f} in')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values_inches.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 0.1
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    if precipitation_max < 2:
        vmin_colorbar, vmax_colorbar = (0, 2)
    elif 2 <= precipitation_max < 4:
        vmin_colorbar, vmax_colorbar = (0, 4)
    else:
        vmin_colorbar, vmax_colorbar = (0, 8)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5), 1))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%.1f', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        country_boundaries = gdf_usa[gdf_usa['admin'] == country]
        if len(country_boundaries) > 0:
            borders = country_boundaries.unary_union.boundary
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        states = gdf_usa[(gdf_usa['admin'] == country) & (gdf_usa['type'] != 'Country')]
        if len(states) > 0:
            states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=4, transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.05, 0.001, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(inches)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Los Angeles': {'lat': 34.05, 'lon': -118.24}, 'San Francisco': {'lat': 37.77, 'lon': -122.42}, 'Seattle': {'lat': 47.61, 'lon': -122.33}, 'Portland': {'lat': 45.52, 'lon': -122.68}, 'San Diego': {'lat': 32.72, 'lon': -117.16}, 'Las Vegas': {'lat': 36.17, 'lon': -115.14}, 'Reno': {'lat': 39.5305, 'lon': -119.813}, 'Phoenix': {'lat': 33.45, 'lon': -112.07}, 'Tucson': {'lat': 32.2566, 'lon': -110.9697}, 'Denver': {'lat': 39.74, 'lon': -104.99}, 'Albuquerque': {'lat': 35.08, 'lon': -106.65}, 'Salt Lake City': {'lat': 40.76, 'lon': -111.89}, 'Houston': {'lat': 29.76, 'lon': -95.37}, 'Dallas': {'lat': 32.78, 'lon': -96.8}, 'San Antonio': {'lat': 29.42, 'lon': -98.49}, 'McAllen': {'lat': 26.2066, 'lon': -98.2325}, 'Amarillo': {'lat': 35.205, 'lon': -101.8369}, 'Oklahoma City': {'lat': 35.4738, 'lon': -97.5205}, 'Miami': {'lat': 25.76, 'lon': -80.27}, 'Orlando': {'lat': 28.5308, 'lon': -81.3825}, 'Jacksonville': {'lat': 30.3316, 'lon': -81.6577}, 'Atlanta': {'lat': 33.75, 'lon': -84.39}, 'New Orleans': {'lat': 29.95, 'lon': -90.07}, 'Nashville': {'lat': 36.16, 'lon': -86.78}, 'Charlotte': {'lat': 35.23, 'lon': -80.84}, 'New York': {'lat': 40.71, 'lon': -74.01}, 'Boston': {'lat': 42.36, 'lon': -71.06}, 'Washington DC': {'lat': 38.91, 'lon': -77.04}, 'Chicago': {'lat': 41.88, 'lon': -87.63}, 'Detroit': {'lat': 42.33, 'lon': -83.1}, 'Minneapolis': {'lat': 44.98, 'lon': -93.27}, 'Kansas City': {'lat': 39.1, 'lon': -94.58}, 'Wichita': {'lat': 37.6916, 'lon': -97.3286}, 'Omaha': {'lat': 41.2591, 'lon': -95.9347}, 'Memphis': {'lat': 35.1402, 'lon': -90.0355}, 'Cincinati': {'lat': 39.1058, 'lon': -84.5141}, 'St. Louis': {'lat': 38.63, 'lon': -90.2}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.12, name, fontsize=7.4, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.0, 0.12, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f'Logo not found en {DEFAULT_LOGO_PATH}')
    ax.text(0.97, 0.1, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteOcean', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/united_states_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/united_states_multimodel_precipitation_map.png')
    plt.show()
    return (fig, ax)

def plot_florida_precipitation(precipitation, admin_boundaries=None, florida_counties=None, show_logo=True):
    """
    Plot accumulated precipitation for Florida in inches.
    Dynamically adjusts the color range from the maximum precipitation value.
    Precipitation is shown over land and ocean.

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with administrative boundaries
    condados_florida : GeoDataFrame
        GeoDataFrame con condados de Florida (opcional)
    mostrar_logo : bool
        Whether to show the logo (default: True)
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    gdf_florida = admin_boundaries[(admin_boundaries['admin'] == 'United States of America') & (admin_boundaries['name'] == 'Florida')].copy()
    lon_min_deg = -88.5
    lon_max_deg = -77.5
    lat_min = 24.3
    lat_max = 31.5
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f' LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f' LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f" Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max + 1, lat_min - 1), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min - 1, lat_max + 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f' LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f' LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    precip_values_inches = precip_values / 25.4
    print(f' Precipitation: Min={precip_values_inches.min():.2f}, Max={precip_values_inches.max():.2f}, Mean={precip_values_inches.mean():.2f} in')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values_inches.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 0.1
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    if precipitation_max < 2:
        vmin_colorbar, vmax_colorbar = (0, 2)
    elif 2 <= precipitation_max < 4:
        vmin_colorbar, vmax_colorbar = (0, 4)
    else:
        vmin_colorbar, vmax_colorbar = (0, 8)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5), 1))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%.1f', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if len(gdf_florida) > 0:
        borders = gdf_florida.unary_union.boundary
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if florida_counties is not None:
        print(f'✅ Dibujando {len(florida_counties)} condados de Florida')
        florida_counties.boundary.plot(ax=ax, edgecolor='grey', linewidth=0.3, alpha=0.8, zorder=8, transform=ccrs.PlateCarree())
    else:
        print('⚠️ No se proporcionaron condados de Florida')
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.02, 0.05, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(in)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    tick_values = np.arange(0, vmax_colorbar + 0.5, 1)
    cbar.set_ticks(tick_values)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Miami': {'lat': 25.76, 'lon': -80.19}, 'Tampa': {'lat': 27.95, 'lon': -82.46}, 'Orlando': {'lat': 28.54, 'lon': -81.38}, 'Jacksonville': {'lat': 30.33, 'lon': -81.66}, 'Tallahassee': {'lat': 30.44, 'lon': -84.28}, 'Fort Lauderdale': {'lat': 26.12, 'lon': -80.14}, 'West Palm Beach': {'lat': 26.71, 'lon': -80.05}, 'Naples': {'lat': 26.14, 'lon': -81.79}, 'Fort Myers': {'lat': 26.64, 'lon': -81.87}, 'Sarasota': {'lat': 27.34, 'lon': -82.53}, 'Pensacola': {'lat': 30.42, 'lon': -87.22}, 'Gainesville': {'lat': 29.65, 'lon': -82.32}, 'Daytona Beach': {'lat': 29.21, 'lon': -81.02}, 'Key West': {'lat': 24.56, 'lon': -81.78}, 'Panama City': {'lat': 30.16, 'lon': -85.66}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.08, name, fontsize=7.4, color='white', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a5490', edgecolor='white', linewidth=0.8, alpha=0.95))
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.07, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f'Logo not found en {DEFAULT_LOGO_PATH}')
    ax.text(0.9, 0.01, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteOcean', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/florida_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/florida_multimodel_precipitation_map.png')
    plt.show()
    return (fig, ax)

def plot_texas_precipitation(precipitation, admin_boundaries=None, texas_counties=None, show_logo=True):
    """
    Plot accumulated precipitation for Texas in inches.
    Dynamically adjusts the color range from the maximum precipitation value.
    Precipitation is shown over land and ocean.

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with administrative boundaries
    condados_texas : GeoDataFrame
        GeoDataFrame con condados de Texas (opcional)
    mostrar_logo : bool
        Whether to show the logo (default: True)
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    gdf_texas = admin_boundaries[(admin_boundaries['admin'] == 'United States of America') & (admin_boundaries['name'] == 'Texas')].copy()
    lon_min_deg = -108
    lon_max_deg = -92
    lat_min = 25.5
    lat_max = 37
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f'  LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f'  LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f"  Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max + 1, lat_min - 1), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min - 1, lat_max + 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f'  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f'  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    precip_values_inches = precip_values / 25.4
    print(f'  Precipitation: Min={precip_values_inches.min():.2f}, Max={precip_values_inches.max():.2f}, Mean={precip_values_inches.mean():.2f} in')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 500)
    lat_fine = np.linspace(lat.min(), lat.max(), 500)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values_inches.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 0.1
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    print(f'Precipitation maximum: {precipitation_max:.2f} in')
    if precipitation_max < 2:
        vmin_colorbar, vmax_colorbar = (0, 2)
    elif 2 <= precipitation_max < 4:
        vmin_colorbar, vmax_colorbar = (0, 4)
    else:
        vmin_colorbar, vmax_colorbar = (0, 8)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(18, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5), 1))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%.1f', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if len(gdf_texas) > 0:
        borders = gdf_texas.unary_union.boundary
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if texas_counties is not None:
        print(f'✅ Dibujando {len(texas_counties)} condados de Texas')
        texas_counties.boundary.plot(ax=ax, edgecolor='grey', linewidth=0.3, alpha=0.8, zorder=8, transform=ccrs.PlateCarree())
    else:
        print('⚠️ No se proporcionaron condados de Texas')
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.02, 0.05, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(in)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    tick_values = np.arange(0, vmax_colorbar + 0.5, 1)
    cbar.set_ticks(tick_values)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Houston': {'lat': 29.76, 'lon': -95.37}, 'San Antonio': {'lat': 29.42, 'lon': -98.49}, 'Dallas': {'lat': 32.78, 'lon': -96.8}, 'Austin': {'lat': 30.27, 'lon': -97.74}, 'El Paso': {'lat': 31.76, 'lon': -106.49}, 'Corpus Christi': {'lat': 27.8, 'lon': -97.4}, 'Laredo': {'lat': 27.51, 'lon': -99.51}, 'Lubbock': {'lat': 33.58, 'lon': -101.86}, 'Amarillo': {'lat': 35.22, 'lon': -101.83}, 'Brownsville': {'lat': 25.9, 'lon': -97.5}, 'Odessa': {'lat': 31.85, 'lon': -102.37}, 'Beaumont': {'lat': 30.09, 'lon': -94.13}, 'Waco': {'lat': 31.55, 'lon': -97.15}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.14, name, fontsize=9, color='white', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a5490', edgecolor='white', linewidth=0.8, alpha=0.95))
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.07, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f'Logo not found en {DEFAULT_LOGO_PATH}')
    ax.text(0.9, 0.01, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteOcean', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/texas_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/texas_multimodel_precipitation_map.png')
    plt.show()
    return (fig, ax)

def plot_mexico_precipitation(precipitation, admin_boundaries=None):
    """
    Plot accumulated precipitation for Mexico in inches.

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with administrative boundaries
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    lon_min_deg = -119
    lon_max_deg = -86
    lat_min = 14
    lat_max = 33
    mexico = admin_boundaries[admin_boundaries['admin'] == 'Mexico'].copy()
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    precip_corrected = precipitation.assign_coords(longitude=lon_corrected)
    margen_lon = 2
    margen_lat = 2
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max + margen_lat, lat_min - margen_lat), longitude=slice(lon_min_deg - margen_lon, lon_max_deg + margen_lon))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min - margen_lat, lat_max + margen_lat), longitude=slice(lon_min_deg - margen_lon, lon_max_deg + margen_lon))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    if precip_area.values.ndim == 3:
        precip_values = precip_area.values[0, :, :]
    else:
        precip_values = precip_area.values
    precip_values_inches = precip_values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values_inches.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 1
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    if precipitation_max < 50:
        vmin_colorbar, vmax_colorbar = (1, 50)
    elif 50 <= precipitation_max < 100:
        vmin_colorbar, vmax_colorbar = (0, 100)
    elif 100 <= precipitation_max < 150:
        vmin_colorbar, vmax_colorbar = (0, 200)
    else:
        vmin_colorbar, vmax_colorbar = (1, 300)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(15, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    buffer_lat = 1
    ax.set_extent([lon_min_deg, lon_max_deg + buffer_lat, lat_min - buffer_lat, lat_max + buffer_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree())
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.05, 0.001, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(mm)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    mexico_geom = mexico.unary_union
    ax.add_geometries([mexico_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=5, alpha=0.8, zorder=5, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)])
    ax.add_geometries([mexico_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='white', linewidth=2, zorder=8, path_effects=[PathEffects.Normal()])
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
                ax.add_geometries([interior_lines], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4, alpha=0.8, zorder=9)
    except Exception as e:
        print(f'Error drawing interior boundaries: {e}')
    major_cities = {'Mexicali': {'state': 'Baja California', 'lat': 32.636, 'lon': -115.475}, 'La Paz': {'state': 'Baja California Sur', 'lat': 24.142, 'lon': -110.313}, 'Hermosillo': {'state': 'Sonora', 'lat': 29.075, 'lon': -110.958}, 'Culiacán': {'state': 'Sinaloa', 'lat': 24.79, 'lon': -107.387}, 'Tepic': {'state': 'Nayarit', 'lat': 21.5, 'lon': -104.9}, 'Guadalajara': {'state': 'Jalisco', 'lat': 20.666, 'lon': -103.391}, 'Cd de Colima': {'state': 'Colima', 'lat': 19.1, 'lon': -103.9}, 'Acapulco': {'state': 'Guerrero', 'lat': 16.862, 'lon': -99.887}, 'Cd de Oaxaca': {'state': 'Oaxaca', 'lat': 17.06, 'lon': -96.723}, 'Salina Cruz': {'state': 'Oaxaca', 'lat': 16.2, 'lon': -95.195}, 'Tuxtla': {'state': 'Chiapas', 'lat': 16.759, 'lon': -93.113}, 'Villahermosa': {'state': 'Tabasco', 'lat': 17.986, 'lon': -92.93}, 'Cd de Campeche': {'state': 'Campeche', 'lat': 19.843, 'lon': -90.525}, 'Mérida': {'state': 'Yucatán', 'lat': 20.975, 'lon': -89.616}, 'Cancún': {'state': 'Quintana Roo', 'lat': 21.174, 'lon': -86.846}, 'Chetumal': {'state': 'Quintana Roo', 'lat': 18.514, 'lon': -88.303}, 'Cd de Veracruz': {'state': 'Veracruz', 'lat': 19.1809, 'lon': -96.142}, 'Poza Rica': {'state': 'Veracruz', 'lat': 20.533, 'lon': -97.459}, 'Cd de México': {'state': 'Ciudad de México', 'lat': 19.428, 'lon': -99.127}, 'León': {'state': 'Guanajuato', 'lat': 21.129, 'lon': -101.673}, 'Cd de Zacatecas': {'state': 'Zacatecas', 'lat': 22.768, 'lon': -102.581}, 'Cd Mante': {'state': 'Tamaulipas', 'lat': 22.743, 'lon': -98.973}, 'Reynosa': {'state': 'Tamaulipas', 'lat': 26.08, 'lon': -98.288}, 'Nuevo Laredo': {'state': 'Tamaulipas', 'lat': 27.41, 'lon': -99.59}, 'Monterrey': {'state': 'Nuevo León', 'lat': 25.675, 'lon': -100.318}, 'Piedras Negras': {'state': 'Coahuila', 'lat': 28.7, 'lon': -100.523}, 'Torreón': {'state': 'Durango', 'lat': 25.543, 'lon': -103.418}, 'Cd de Durango': {'state': 'Durango', 'lat': 24.934, 'lon': -104.911}, 'Cd Juárez': {'state': 'Chihuahua', 'lat': 31.72, 'lon': -106.46}, 'Cd de Chihuahua': {'state': 'Chihuahua', 'lat': 28.635, 'lon': -106.088}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat = metadata['lat']
        city_lon = metadata['lon']
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.18, name, fontsize=8, color='white', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a5490', edgecolor='white', linewidth=0.8, alpha=0.95))
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black', facecolor='none'), linewidth=0.8, alpha=0.3, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.4, edgecolor='black', alpha=0.3, zorder=5)
    ax.add_feature(cfeature.STATES, linewidth=1, edgecolor='black', alpha=0.2, zorder=5)
    try:
        logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower left', bbox_to_anchor=(0.86, 0.04, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print('Logo not found')
    ax.text(0.97, 0.03, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', zorder=35, path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.Normal()])
    output_file = str(DEFAULT_OUTPUT_DIR / 'mexico_precipitation_map.png')
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'Map saved: {output_file}')
    plt.show()
    return (fig, ax)

def plot_regional_precipitation(precipitation, admin_boundaries=None, show_logo=True):
    """
    Plot accumulated precipitation for the full United States, Mexico,
    Central America, Caribbean, Colombia, and Venezuela region.
    Dynamically adjusts the color range from the maximum precipitation value.
    Precipitation is shown over land and ocean.

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with administrative boundaries
    mostrar_logo : bool
        Whether to show the logo (default: True)
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    countries = ['United States of America', 'Mexico', 'Guatemala', 'Belize', 'Honduras', 'El Salvador', 'Nicaragua', 'Costa Rica', 'Panama', 'Colombia', 'Venezuela', 'Cuba', 'Haiti', 'Dominican Republic', 'Jamaica', 'Puerto Rico', 'Trinidad and Tobago', 'Bahamas']
    gdf_regional = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg = -125
    lon_max_deg = -58
    lat_min = 0.5
    lat_max = 50
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f' LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f' LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f" Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max + 1, lat_min - 1), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min - 1, lat_max + 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f' LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f' LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f' Precipitation: Min={precip_values.min():.2f}, Max={precip_values.max():.2f}, Mean={precip_values.mean():.2f} mm')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 600)
    lat_fine = np.linspace(lat.min(), lat.max(), 600)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 5
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    if precipitation_max < 50:
        vmin_colorbar, vmax_colorbar = (1, 50)
    elif 50 <= precipitation_max < 100:
        vmin_colorbar, vmax_colorbar = (0, 100)
    elif 100 <= precipitation_max < 150:
        vmin_colorbar, vmax_colorbar = (0, 200)
    else:
        vmin_colorbar, vmax_colorbar = (1, 300)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(22, 14))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        country_boundaries = gdf_regional[gdf_regional['admin'] == country]
        if len(country_boundaries) > 0:
            borders = country_boundaries.unary_union.boundary
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.0, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.0, path_effects=[PathEffects.Normal()], zorder=7)
    countries_with_states = ['United States of America', 'Mexico', 'Colombia', 'Venezuela']
    for country in countries_with_states:
        states = gdf_regional[(gdf_regional['admin'] == country) & (gdf_regional['type'] != 'Country')]
        if len(states) > 0:
            states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=4, transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.05, 0.001, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(mm)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.plot(city_lon, city_lat, 'o', color='red', markersize=3.5, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(city_lon, city_lat - 0.25, name, fontsize=6.5, color='white', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a5490', edgecolor='white', linewidth=0.8, alpha=0.95))
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_LOGO_PATH)
            axins_logo = inset_axes(ax, width='7%', height='7%', loc='lower right', bbox_to_anchor=(-0.9, 0.18, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print(f'Logo not found en {DEFAULT_LOGO_PATH}')
    ax.text(0.15, 0.3, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.4, 0.04, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.21, 0.03, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8.5, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.21, 0.018, 'Created by ElTiempoconLorenzo', transform=ax.transAxes, fontsize=7, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/regional_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/regional_multimodel_precipitation_map.png')
    plt.show()
    return (fig, ax)

def plot_iberia_precipitation(precipitation, admin_boundaries=None):
    """
    Plot accumulated precipitation for Iberia (Spain and Portugal) in millimeters.

    Parameters:
    -----------
    precip_dataarray : xarray.DataArray
        DataArray with precipitation in mm and coordinates 'longitude' y 'latitude'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with Spanish autonomous communities and Portuguese regions
    """
    if admin_boundaries is None:
        raise ValueError("You must provide admin_boundaries.")
    lon_min_deg = -12
    lon_max_deg = 5
    lat_min = 35
    lat_max = 45
    espana = admin_boundaries[admin_boundaries['admin'] == 'Spain'].copy()
    portugal = admin_boundaries[admin_boundaries['admin'] == 'Portugal'].copy()
    iberia = pd.concat([espana, portugal])
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    precip_corrected = precipitation.assign_coords(longitude=lon_corrected)
    margen_lon = 2
    margen_lat = 2
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max + margen_lat, lat_min - margen_lat), longitude=slice(lon_min_deg - margen_lon, lon_max_deg + margen_lon))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min - margen_lat, lat_max + margen_lat), longitude=slice(lon_min_deg - margen_lon, lon_max_deg + margen_lon))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    if precip_area.values.ndim == 3:
        precip_values = precip_area.values[0, :, :]
    else:
        precip_values = precip_area.values
    precip_vals_mm = precip_values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_vals_mm.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 3
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_masked_fine)
    if precipitation_max < 50:
        vmin_colorbar, vmax_colorbar = (1, 50)
    elif 50 <= precipitation_max < 100:
        vmin_colorbar, vmax_colorbar = (0, 100)
    elif 100 <= precipitation_max < 150:
        vmin_colorbar, vmax_colorbar = (0, 200)
    else:
        vmin_colorbar, vmax_colorbar = (1, 300)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree())
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    cax = inset_axes(ax, width='3%', height='40%', loc='lower left', bbox_to_anchor=(0.02, 0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label('(L/m²)', fontsize=11, weight='bold')
    label_obj = cbar.ax.yaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    espana_geom = espana.unary_union
    ax.add_geometries([espana_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=5, alpha=0.8, zorder=5, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)])
    ax.add_geometries([espana_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='white', linewidth=2, zorder=8, path_effects=[PathEffects.Normal()])
    portugal_geom = portugal.unary_union
    ax.add_geometries([portugal_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=5, alpha=0.8, zorder=5, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)])
    ax.add_geometries([portugal_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='white', linewidth=2, zorder=8, path_effects=[PathEffects.Normal()])
    try:
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
                ax.add_geometries([interior_lines_esp], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4, alpha=0.8, zorder=9)
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
                ax.add_geometries([interior_lines_pt], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4, alpha=0.8, zorder=9)
    except Exception as e:
        print(f'Error drawing interior boundaries: {e}')
    regional_capitals = {'Ceuta': {'lat': 35.8894, 'lon': -5.3213}, 'Melilla': {'lat': 35.2943, 'lon': -2.9526}, 'Álava': {'lat': 42.85, 'lon': -2.69}, 'Albacete': {'lat': 38.99, 'lon': -1.86}, 'Alicante': {'lat': 38.35, 'lon': -0.48}, 'Almería': {'lat': 36.84, 'lon': -2.46}, 'Asturias': {'lat': 43.37, 'lon': -5.86}, 'Ávila': {'lat': 40.65, 'lon': -4.7}, 'Badajoz': {'lat': 38.88, 'lon': -6.97}, 'Barcelona': {'lat': 41.39, 'lon': 2.17}, 'Burgos': {'lat': 42.34, 'lon': -3.7}, 'Cáceres': {'lat': 39.48, 'lon': -6.37}, 'Cádiz': {'lat': 36.53, 'lon': -6.29}, 'Cantabria': {'lat': 43.46, 'lon': -3.81}, 'Castellón': {'lat': 39.98, 'lon': -0.04}, 'Ciudad Real': {'lat': 38.99, 'lon': -3.93}, 'Córdoba': {'lat': 37.88, 'lon': -4.78}, 'Cuenca': {'lat': 40.07, 'lon': -2.13}, 'Girona': {'lat': 41.98, 'lon': 2.82}, 'Granada': {'lat': 37.18, 'lon': -3.6}, 'Guadalajara': {'lat': 40.63, 'lon': -3.17}, 'Guipúzkoa': {'lat': 43.31, 'lon': -1.98}, 'Huelva': {'lat': 37.26, 'lon': -6.95}, 'Huesca': {'lat': 42.14, 'lon': -0.41}, 'Ibiza': {'lat': 38.9105, 'lon': 1.4247}, 'Mallorca': {'lat': 39.61, 'lon': 2.97}, 'Menorca': {'lat': 40.0002, 'lon': 3.84}, 'Jaén': {'lat': 37.77, 'lon': -3.79}, 'La Coruña': {'lat': 43.36, 'lon': -8.41}, 'La Rioja': {'lat': 42.29, 'lon': -2.54}, 'León': {'lat': 42.6, 'lon': -5.57}, 'Lérida': {'lat': 41.61, 'lon': 0.62}, 'Lugo': {'lat': 43.01, 'lon': -7.56}, 'Madrid': {'lat': 40.42, 'lon': -3.7}, 'Málaga': {'lat': 36.72, 'lon': -4.42}, 'Murcia': {'lat': 37.99, 'lon': -1.13}, 'Navarra': {'lat': 42.69, 'lon': -1.68}, 'Orense': {'lat': 42.34, 'lon': -7.86}, 'Palencia': {'lat': 42.01, 'lon': -4.53}, 'Pontevedra': {'lat': 42.43, 'lon': -8.64}, 'Salamanca': {'lat': 40.97, 'lon': -5.66}, 'Segovia': {'lat': 40.95, 'lon': -4.12}, 'Sevilla': {'lat': 37.39, 'lon': -5.99}, 'Soria': {'lat': 41.77, 'lon': -2.47}, 'Tarragona': {'lat': 41.12, 'lon': 1.25}, 'Teruel': {'lat': 40.34, 'lon': -1.11}, 'Toledo': {'lat': 39.86, 'lon': -4.03}, 'Valencia': {'lat': 39.47, 'lon': -0.38}, 'Valladolid': {'lat': 41.65, 'lon': -4.72}, 'Vizkaya': {'lat': 43.26, 'lon': -2.93}, 'Zamora': {'lat': 41.5, 'lon': -5.74}, 'Zaragoza': {'lat': 41.65, 'lon': -0.89}, 'Lisboa': {'lat': 38.722, 'lon': -9.139}, 'Porto': {'lat': 41.158, 'lon': -8.629}, 'Coimbra': {'lat': 40.211, 'lon': -8.429}, 'Évora': {'lat': 38.571, 'lon': -7.907}, 'Faro': {'lat': 37.017, 'lon': -7.93}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in regional_capitals.items():
        city_lat = metadata['lat']
        city_lon = metadata['lon']
        if not (lon_min_deg <= city_lon <= lon_max_deg and lat_min <= city_lat <= lat_max):
            continue
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.text(city_lon, city_lat, name, fontsize=9, color='white', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a5490', edgecolor='white', linewidth=0.8, alpha=0.95))
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black', facecolor='none'), linewidth=0.8, alpha=0.3, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.4, edgecolor='black', alpha=0.3, zorder=5)
    ax.add_feature(cfeature.STATES, linewidth=1, edgecolor='black', alpha=0.2, zorder=5)
    try:
        logo_img = mpimg.imread(DEFAULT_SPAIN_LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower left', bbox_to_anchor=(0.86, 0.04, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print('Logo not found (opcional)')
    ax.text(0.97, 0.03, 'Created by MeteoSpain', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', zorder=35, path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.4, 0.04, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.21, 0.03, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8.5, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.21, 0.015, 'Created by MeteoSpain', transform=ax.transAxes, fontsize=7, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_file = str(DEFAULT_OUTPUT_DIR / 'iberia_precipitation_map.png')
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'Map saved: {output_file}')
    plt.show()
    return (fig, ax)

def plot_canary_islands_precipitation(precipitation, canary_islands=None, admin_boundaries=None, show_logo=True):
    """
    Plot accumulated precipitation for the Canary Islands and save the figure to outputs/maps.
    Dynamically adjusts the color range from the maximum precipitation value.
    """
    lon_min_deg, lon_max_deg = (-19.5, -12)
    lat_min, lat_max = (26.5, 30.5)
    if 'longitude' not in precipitation.coords or 'latitude' not in precipitation.coords:
        raise ValueError("Input DataArray must have 'longitude' and 'latitude' coordinates.")
    lon_orig = precipitation.longitude.values
    lat_orig = precipitation.latitude.values
    print(f'Original coordinates:')
    print(f'  LON: [{lon_orig.min():.2f}, {lon_orig.max():.2f}] - {len(lon_orig)} points')
    print(f'  LAT: [{lat_orig.min():.2f}, {lat_orig.max():.2f}] - {len(lat_orig)} points')
    print(f"  Latitude order: {('Descending (N to S)' if lat_orig[0] > lat_orig[-1] else 'Ascending (S to N)')}")
    precip_corrected = precipitation
    if lat_orig[0] > lat_orig[-1]:
        precip_area = precip_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    else:
        precip_area = precip_corrected.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min_deg, lon_max_deg))
    lat = precip_area.latitude.values
    lon = precip_area.longitude.values
    print(f'\nSelected area:')
    print(f'  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f'  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty. Check the bounds: LON[{lon_min_deg},{lon_max_deg}], LAT[{lat_min},{lat_max}]')
    precip_values = precip_area.values[0, :, :] if precip_area.values.ndim == 3 else precip_area.values
    print(f'  Precipitation: Min={precip_values.min():.2f}, Max={precip_values.max():.2f}, Mean={precip_values.mean():.2f} mm')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = precip_values.ravel()
    precip_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='cubic')
    precip_fine_smooth = gaussian_filter(precip_fine, sigma=2)
    umbral_precip = 1
    precip_masked_fine = np.where(precip_fine_smooth > umbral_precip, precip_fine_smooth, np.nan)
    precipitation_max = np.nanmax(precip_fine_smooth)
    print(precipitation_max)
    if precipitation_max < 50:
        vmin_colorbar, vmax_colorbar = (1, 50)
    elif 50 <= precipitation_max < 100:
        vmin_colorbar, vmax_colorbar = (0, 100)
    else:
        vmin_colorbar, vmax_colorbar = (0, 150)
    levels = np.linspace(vmin_colorbar, vmax_colorbar, 20)
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    if canary_islands is None and admin_boundaries is not None:
        canary_islands = admin_boundaries[admin_boundaries['admin'] == 'Spain'].copy()
    elif canary_islands is None:
        raise ValueError("You must provide either canary_islands or admin_boundaries.")
    canary_islands.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=5)
    colors_precip = ['#b6ffb6', '#66ff66', '#00cc00', '#006400', '#ffff00', '#ffb300', '#ff6600', '#ff0000', '#d00070', '#a000c0', '#6a0dad']
    cmap_precip = mcolors.LinearSegmentedColormap.from_list('precipitation', colors_precip)
    cf = ax.contourf(lon2d_fine, lat2d_fine, precip_masked_fine, levels=levels, cmap=cmap_precip, extend='max', transform=ccrs.PlateCarree(), zorder=3)
    contour_levels = np.unique(np.round(np.linspace(max(umbral_precip, vmin_colorbar), vmax_colorbar, 5)).astype(int))
    cs = ax.contour(lon2d_fine, lat2d_fine, precip_fine_smooth, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
    labels = ax.clabel(cs, inline=True, fontsize=7.5, fmt='%d', inline_spacing=8)
    for label in labels:
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.0, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(L/m²)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=9, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Tenerife': {'lat': 28.3191, 'lon': -16.27}, 'Gran Canaria': {'lat': 27.7605, 'lon': -15.5838}, 'Lanzarote': {'lat': 28.9394, 'lon': -13.5316}, 'Fuerteventura': {'lat': 28.06, 'lon': -13.9113}, 'La Palma': {'lat': 28.4647, 'lon': -17.8391}, 'La Gomera': {'lat': 28.0263, 'lon': -17.2322}, 'El Hierro': {'lat': 27.6433, 'lon': -17.9783}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    precip_values = precip_fine_smooth.ravel()
    for name, metadata in major_cities.items():
        city_lat, city_lon = (metadata['lat'], metadata['lon'])
        _, idx = tree.query([city_lat, city_lon])
        precipitation_value = precip_values[idx]
        ax.text(city_lon, city_lat - 0.08, name, fontsize=7.4, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(DEFAULT_SPAIN_LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.05, 0.03, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print('Logo not found')
    ax.text(1.0, 0.15, 'Created by MeteoSpain', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    banner_box = FancyBboxPatch((0.01, 0.01), 0.48, 0.055, boxstyle='round,pad=0.005', transform=ax.transAxes, facecolor='white', edgecolor='#2C3E50', linewidth=1.5, alpha=0.95, zorder=20)
    ax.add_patch(banner_box)
    ax.text(0.25, 0.0375, 'Experimental Hybrid Multi-Model (Physics + AI)', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='#2C3E50', fontweight='bold', zorder=21)
    ax.text(0.25, 0.02, 'Created by MeteoSpain', transform=ax.transAxes, fontsize=6.5, ha='center', va='center', color='#34495E', fontstyle='italic', fontweight='semibold', zorder=21)
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/canary_islands_multimodel_precipitation_map.png', dpi=800, bbox_inches='tight')
    plt.show()
    return (fig, ax)
