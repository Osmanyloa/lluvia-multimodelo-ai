import os
import numpy as np
import xarray as xr
import cfgrib
from ecmwf.opendata import Client
from scipy.ndimage import zoom, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Instalar PyVista si no está disponible
try:
    import pyvista as pv
    print("✅ PyVista disponible")
except ImportError:
    print("📦 Instalando PyVista...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyvista', 'trame'])
    import pyvista as pv
    print("✅ PyVista instalado")

# Detectar si estamos en Colab/Jupyter
try:
    import google.colab
    EN_COLAB = True
    pv.set_jupyter_backend('static')  # Mejor para Colab
    print("✅ Modo: Google Colab")
except:
    EN_COLAB = False
    try:
        get_ipython()
        pv.set_jupyter_backend('trame')  # Interactivo para Jupyter
        print("✅ Modo: Jupyter Notebook")
    except:
        pv.set_jupyter_backend('static')  # Estático para scripts
        print("✅ Modo: Script Python")

# --- CONFIGURACIÓN ---
fecha = "20251027"
corrida = "18"
lead_time = 72
carpeta = "./datos_ecmwf"
os.makedirs(carpeta, exist_ok=True)

# --- REGIÓN CARIBE ---
lon_min, lon_max = -95, -55
lat_min, lat_max = 5, 30

# --- PARÁMETROS DE VISUALIZACIÓN ---
z_scale_terrain = 0.020     # Exageración vertical del terreno (aumentado)
z_scale_clouds = 0.001      # Escala vertical de las nubes
z_offset_clouds = 1.5       # Altura base de las nubes (reducido para mejor vista)
resolution = 0.4            # Resolución en grados (menor = más detalle)

print("=" * 70)
print("VISUALIZACIÓN 3D VOLUMÉTRICA: NUBES + TOPOGRAFÍA")
print("=" * 70)
print(f"📍 Región: Caribe ({lon_min}°E a {lon_max}°E, {lat_min}°N a {lat_max}°N)")
print(f"📅 Fecha: {fecha}, Corrida: {corrida}Z, Lead time: +{lead_time}h")

# =============================================================================
# PASO 1: DESCARGAR DATOS DE NUBES
# =============================================================================
print("\n☁️  DESCARGANDO DATOS DE NUBES...")

archivo_nubes = f"ifs_sfc_clouds_{fecha}_{corrida}z_lead{lead_time}h.grib"
ruta_nubes = os.path.join(carpeta, archivo_nubes)

if not os.path.exists(ruta_nubes):
    try:
        client = Client(source="ecmwf")

        client.retrieve(
            date=int(fecha),
            time=int(corrida),
            step=lead_time,
            stream="oper",
            type="fc",
            levtype="sfc",
            param=["tcc", "lcc", "mcc", "hcc"],  # Total, Low, Medium, High cloud cover
            target=ruta_nubes
        )
        print(f"✅ Datos descargados: {ruta_nubes}")

    except Exception as e:
        print(f"⚠️  Error: {e}")
        print("💡 Usando datos sintéticos...")
        ruta_nubes = None
else:
    print(f"✅ Archivo encontrado: {ruta_nubes}")

# =============================================================================
# PASO 2: GENERAR TOPOGRAFÍA REALISTA
# =============================================================================
print("\n🏔️  GENERANDO TOPOGRAFÍA DEL CARIBE...")

def crear_topografia_caribe(lons, lats):
    """Genera topografía realista del Caribe con montañas y océano"""
    lon2d, lat2d = np.meshgrid(lons, lats)
    elevation = np.zeros_like(lon2d)

    # Cordillera de los Andes (Venezuela/Colombia)
    andes = ((lon2d > -75) & (lon2d < -65) & (lat2d > 5) & (lat2d < 12))
    elevation += andes * 4500 * np.exp(-((lon2d + 70)**2 + (lat2d - 8)**2) / 40)

    # Sierra Madre (México)
    mexico = ((lon2d > -105) & (lon2d < -95) & (lat2d > 15) & (lat2d < 25))
    elevation += mexico * 3500 * (1 + 0.4 * np.sin(lat2d * 1.5))

    # Centroamérica
    central = ((lon2d > -92) & (lon2d < -80) & (lat2d > 8) & (lat2d < 18))
    elevation += central * 2800 * (1 + 0.3 * np.cos(lon2d / 3))

    # Montañas de Cuba
    cuba = ((lon2d > -85) & (lon2d < -74) & (lat2d > 19.5) & (lat2d < 23))
    elevation += cuba * 2000 * np.exp(-((lat2d - 21)**2) / 3)

    # Hispaniola (Rep. Dominicana/Haití)
    hispaniola = ((lon2d > -75) & (lon2d < -68) & (lat2d > 17.5) & (lat2d < 20))
    elevation += hispaniola * 3100 * np.exp(-((lon2d + 71)**2 + (lat2d - 19)**2) / 5)

    # Jamaica
    jamaica = ((lon2d > -78.5) & (lon2d < -76) & (lat2d > 17.5) & (lat2d < 18.5))
    elevation += jamaica * 2200

    # Puerto Rico
    puerto_rico = ((lon2d > -67.5) & (lon2d < -65) & (lat2d > 17.8) & (lat2d < 18.6))
    elevation += puerto_rico * 1300

    # Textura/rugosidad
    np.random.seed(42)
    noise = np.random.randn(*elevation.shape) * 80
    elevation += gaussian_filter(noise, sigma=1.5)

    # Océano (ligeramente negativo)
    elevation = np.maximum(elevation, -200)

    return elevation

lons = np.arange(lon_min, lon_max + resolution, resolution)
lats = np.arange(lat_min, lat_max + resolution, resolution)

elevation = crear_topografia_caribe(lons, lats)
print(f"   Grid: {len(lons)} x {len(lats)} puntos")
print(f"   Elevación: min={elevation.min():.0f}m, max={elevation.max():.0f}m")

# =============================================================================
# PASO 3: PROCESAR DATOS DE NUBES
# =============================================================================
print("\n☁️  PROCESANDO ESTRUCTURA DE NUBES...")

cloud_height = None

if ruta_nubes and os.path.exists(ruta_nubes):
    try:
        ds_nubes = cfgrib.open_dataset(ruta_nubes)

        lcc = ds_nubes['lcc'].values if 'lcc' in ds_nubes else None
        mcc = ds_nubes['mcc'].values if 'mcc' in ds_nubes else None
        hcc = ds_nubes['hcc'].values if 'hcc' in ds_nubes else None

        lons_nubes = ds_nubes['longitude'].values
        lats_nubes = ds_nubes['latitude'].values

        lon_mask = (lons_nubes >= lon_min) & (lons_nubes <= lon_max)
        lat_mask = (lats_nubes >= lat_min) & (lats_nubes <= lat_max)

        cloud_height = np.zeros((len(lats), len(lons)))

        if lcc is not None:
            lcc_crop = lcc[np.ix_(lat_mask, lon_mask)]
            lcc_interp = zoom(lcc_crop, (len(lats)/lcc_crop.shape[0], len(lons)/lcc_crop.shape[1]))
            cloud_height += lcc_interp * 1500

        if mcc is not None:
            mcc_crop = mcc[np.ix_(lat_mask, lon_mask)]
            mcc_interp = zoom(mcc_crop, (len(lats)/mcc_crop.shape[0], len(lons)/mcc_crop.shape[1]))
            cloud_height += mcc_interp * 4000

        if hcc is not None:
            hcc_crop = hcc[np.ix_(lat_mask, lon_mask)]
            hcc_interp = zoom(hcc_crop, (len(lats)/hcc_crop.shape[0], len(lons)/hcc_crop.shape[1]))
            cloud_height += hcc_interp * 12000

        print(f"   ✅ Altura nubes: min={cloud_height.min():.0f}m, max={cloud_height.max():.0f}m")

    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        cloud_height = None

# Generar nubes sintéticas si no hay datos reales
if cloud_height is None:
    print("   Generando nubes sintéticas realistas...")
    lon2d, lat2d = np.meshgrid(lons, lats)
    cloud_height = np.zeros_like(lon2d)

    # Nubes convectivas (torres sobre tierra caliente)
    convective = (elevation > 800) & (np.random.rand(*lon2d.shape) > 0.55)
    cloud_height += convective * (9000 + 6000 * np.random.rand(*lon2d.shape))

    # Nubes estratiformes (capas sobre océano)
    stratiform = (elevation < 100) & (np.random.rand(*lon2d.shape) > 0.4)
    cloud_height += stratiform * (2500 + 2000 * np.random.rand(*lon2d.shape))

    # Nubes orográficas (se forman en montañas)
    orographic = (elevation > 1500)
    cloud_height += orographic * (elevation * 0.6 + 3000 * np.random.rand(*lon2d.shape))

    # Sistemas frontales (bandas de nubes)
    for i in range(3):
        center_lat_cloud = np.random.uniform(lat_min + 5, lat_max - 5)
        frontal = np.exp(-((lat2d - center_lat_cloud)**2) / 10) * (np.random.rand(*lon2d.shape) > 0.3)
        cloud_height += frontal * (5000 + 4000 * np.random.rand(*lon2d.shape))

    cloud_height = gaussian_filter(cloud_height, sigma=2)
    print(f"   ✅ Nubes sintéticas: min={cloud_height.min():.0f}m, max={cloud_height.max():.0f}m")

# =============================================================================
# PASO 4: CREAR VISUALIZACIÓN 3D VOLUMÉTRICA CON PYVISTA
# =============================================================================
print("\n🎨 GENERANDO VISUALIZACIÓN 3D VOLUMÉTRICA...")

# Crear plotter
plotter = pv.Plotter(window_size=[1920, 1080], off_screen=True)
plotter.set_background('#0a0f19')  # Azul oscuro espacial

# Preparar datos
lon2d, lat2d = np.meshgrid(lons, lats)

# Calcular centro de la región
center_lon = np.mean([lon_min, lon_max])
center_lat = np.mean([lat_min, lat_max])

# Initialize center_z with a base value, to ensure it's always defined
center_z = z_offset_clouds

# Calculate center_z based on cloud_height, if available and valid
if cloud_height is not None and cloud_height.size > 0:
    center_z = z_offset_clouds + (cloud_height.max() * z_scale_clouds) / 2
else:
    print("⚠️  cloud_height is not valid or empty, using default center_z for camera.")
    # If no clouds, or invalid, set center_z to be a bit above the max terrain
    center_z = z_offset_clouds + (elevation.max() * z_scale_terrain) / 2 + 1

# --- CAPA 0: MAPA BASE PLANO (PAÍSES Y OCÉANOS) ---
print("   Creando mapa base con países y océanos...")

# Crear superficie plana en z=0 para el mapa base
z_base = np.zeros_like(lon2d) - 0.5  # Ligeramente debajo del nivel del mar

points_base = np.c_[lon2d.ravel(), lat2d.ravel(), z_base.ravel()]

# Crear malla de cuadrados
faces_base = []
for i in range(len(lats) - 1):
    for j in range(len(lons) - 1):
        idx = i * len(lons) + j
        faces_base.extend([4, idx, idx + 1, idx + len(lons) + 1, idx + len(lons)])

faces_base = np.array(faces_base)
base_mesh = pv.PolyData(points_base, faces_base)

# Crear textura de mapa base (océano/tierra)
map_colors = np.zeros((*elevation.shape, 3))

# Océanos en azul
ocean_mask = elevation < 10
map_colors[ocean_mask] = [0.15, 0.35, 0.65]  # Azul océano

# Tierra en verde/marrón según elevación
land_mask = ~ocean_mask
elevation_land = elevation[land_mask]
if len(elevation_land) > 0:
    elev_norm = (elevation_land - elevation_land.min()) / (elevation_land.max() - elevation_land.min() + 1e-10)
    map_colors[land_mask, 0] = 0.4 + 0.3 * elev_norm  # R
    map_colors[land_mask, 1] = 0.6 - 0.2 * elev_norm  # G
    map_colors[land_mask, 2] = 0.3 - 0.2 * elev_norm  # B

base_mesh['rgb'] = (map_colors.reshape(-1, 3) * 255).astype(np.uint8)

plotter.add_mesh(
    base_mesh,
    scalars='rgb',
    rgb=True,
    lighting=True,
    ambient=0.5,
    diffuse=0.5,
    show_scalar_bar=False,
    opacity=1.0
)

# --- AÑADIR LÍNEAS DE COSTA Y FRONTERAS ---
print("   Dibujando costas y fronteras...")

try:
    import cartopy.feature as cfeature
    from cartopy.feature import COASTLINE, BORDERS

    # Dibujar líneas de costa
    for geom in COASTLINE.geometries():
        try:
            if geom.geom_type == 'LineString':
                coords = np.array(geom.coords)
                lons_costa = coords[:, 0]
                lats_costa = coords[:, 1]

                mask = ((lons_costa >= lon_min - 2) & (lons_costa <= lon_max + 2) &
                        (lats_costa >= lat_min - 2) & (lats_costa <= lat_max + 2))

                if np.any(mask):
                    lons_seg = lons_costa[mask]
                    lats_seg = lats_costa[mask]
                    z_seg = np.zeros_like(lons_seg) - 0.3

                    points = np.c_[lons_seg, lats_seg, z_seg]
                    if len(points) > 1:
                        line = pv.lines_from_points(points)
                        plotter.add_mesh(line, color='black', line_width=3, opacity=1.0)

            elif geom.geom_type == 'MultiLineString':
                for subgeom in geom.geoms:
                    coords = np.array(subgeom.coords)
                    lons_costa = coords[:, 0]
                    lats_costa = coords[:, 1]

                    mask = ((lons_costa >= lon_min - 2) & (lons_costa <= lon_max + 2) &
                            (lats_costa >= lat_min - 2) & (lats_costa <= lat_max + 2))

                    if np.any(mask):
                        lons_seg = lons_costa[mask]
                        lats_seg = lats_costa[mask]
                        z_seg = np.zeros_like(lons_seg) - 0.3

                        points = np.c_[lons_seg, lats_seg, z_seg]
                        if len(points) > 1:
                            line = pv.lines_from_points(points)
                            plotter.add_mesh(line, color='black', line_width=3, opacity=1.0)
        except:
            continue

    # Dibujar fronteras
    for geom in BORDERS.geometries():
        try:
            if geom.geom_type == 'LineString':
                coords = np.array(geom.coords)
                lons_front = coords[:, 0]
                lats_front = coords[:, 1]

                mask = ((lons_front >= lon_min - 2) & (lons_front <= lon_max + 2) &
                        (lats_front >= lat_min - 2) & (lats_front <= lat_max + 2))

                if np.any(mask):
                    lons_seg = lons_front[mask]
                    lats_seg = lats_front[mask]
                    z_seg = np.zeros_like(lons_seg) - 0.3

                    points = np.c_[lons_seg, lats_seg, z_seg]
                    if len(points) > 1:
                        line = pv.lines_from_points(points)
                        plotter.add_mesh(line, color='darkgray', line_width=2,
                                       opacity=0.7, style='dashed')

            elif geom.geom_type == 'MultiLineString':
                for subgeom in geom.geoms:
                    coords = np.array(subgeom.coords)
                    lons_front = coords[:, 0]
                    lats_front = coords[:, 1]

                    mask = ((lons_front >= lon_min - 2) & (lons_front <= lon_max + 2) &
                            (lats_front >= lat_min - 2) & (lats_front <= lat_max + 2))

                    if np.any(mask):
                        lons_seg = lons_front[mask]
                        lats_seg = lats_front[mask]
                        z_seg = np.zeros_like(lons_seg) - 0.3

                        points = np.c_[lons_seg, lats_seg, z_seg]
                        if len(points) > 1:
                            line = pv.lines_from_points(points)
                            plotter.add_mesh(line, color='darkgray', line_width=2,
                                           opacity=0.7, style='dashed')
        except:
            continue

    print("   ✅ Costas y fronteras dibujadas")

except Exception as e:
    print(f"   ⚠️  Cartopy no disponible: {e}")
    print("   Continuando sin líneas de costa detalladas...")

# --- SUPERFICIE 1: TERRENO CON RELIEVE ---
print("   Creando superficie del terreno con relieve...")

z_terrain = elevation * z_scale_terrain
points_terrain = np.c_[lon2d.ravel(), lat2d.ravel(), z_terrain.ravel()]

# Estructura de malla
faces = []
for i in range(len(lats) - 1):
    for j in range(len(lons) - 1):
        idx = i * len(lons) + j
        faces.extend([4, idx, idx + 1, idx + len(lons) + 1, idx + len(lons)])

faces = np.array(faces)
terrain_mesh = pv.PolyData(points_terrain, faces)

# Colorear terreno por elevación con transparencia para ver el mapa base
terrain_mesh['elevation'] = elevation.ravel()

# Añadir terreno semitransparente sobre el mapa
plotter.add_mesh(
    terrain_mesh,
    scalars='elevation',
    cmap='terrain',  # Colormap topográfico
    lighting=True,
    smooth_shading=True,
    show_scalar_bar=False,
    ambient=0.4,
    diffuse=0.6,
    specular=0.2,
    specular_power=15,
    opacity=0.85  # Semitransparente para ver mapa base
)

# --- SUPERFICIE 2: NUBES VOLUMÉTRICAS ---
print("   Creando estructura volumétrica de nubes...")

# Máscara de nubes significativas
cloud_mask = cloud_height > 300

# Altura de nubes escalada
z_clouds = z_offset_clouds + cloud_height * z_scale_clouds
z_clouds_masked = np.where(cloud_mask, z_clouds, np.nan)

# Crear múltiples capas de nubes para efecto volumétrico
num_layers = 5
for layer_idx in range(num_layers):
    # Cada capa a diferente altura
    height_factor = 0.3 + (layer_idx / num_layers) * 0.7
    z_layer = z_offset_clouds + cloud_height * z_scale_clouds * height_factor

    # Solo donde hay nubes
    valid_mask = cloud_height > (300 * (1 + layer_idx * 0.5))
    z_layer_masked = np.where(valid_mask, z_layer, np.nan)

    points_layer = np.c_[lon2d.ravel(), lat2d.ravel(), z_layer_masked.ravel()]

    # Eliminar puntos inválidos
    valid_points = ~np.isnan(points_layer[:, 2])
    if not np.any(valid_points):
        continue

    points_layer = points_layer[valid_points]
    cloud_data = pv.PolyData(points_layer)

    # Crear superficie con Delaunay 2D
    try:
        cloud_surf = cloud_data.delaunay_2d()

        # Altura de nubes como escalar
        cloud_heights_layer = cloud_height.ravel()[valid_points]
        cloud_surf['cloud_height'] = cloud_heights_layer

        # Opacidad variable por capa y altura
        opacity_base = 0.3 - (layer_idx * 0.04)

        plotter.add_mesh(
            cloud_surf,
            scalars='cloud_height',
            cmap='Blues_r',  # Azul invertido (blanco arriba, azul abajo)
            clim=[0, 15000],
            opacity=opacity_base,
            lighting=True,
            smooth_shading=True,
            show_scalar_bar=False,
            ambient=0.4,
            diffuse=0.6,
            specular=0.8,
            specular_power=100
        )
    except:
        pass

print("   ✅ Nubes volumétricas creadas")

# --- AÑADIR ILUMINACIÓN DRAMÁTICA ---
print("   Configurando iluminación cinematográfica...")

# Luz principal (sol desde el lado)
light1 = pv.Light(position=(lon_max + 30, lat_min - 10, 40),
                  focal_point=(center_lon, center_lat, 5),
                  color='#FFF8DC', intensity=1.0)  # Luz cálida
plotter.add_light(light1)

# Luz de relleno (desde el frente)
light2 = pv.Light(position=(lon_min - 20, lat_max + 10, 25),
                  focal_point=(center_lon, center_lat, 3),
                  color='lightblue', intensity=0.4)
plotter.add_light(light2)

# Luz de respaldo (desde atrás, sutil)
light3 = pv.Light(position=(center_lon, center_lat - 40, 15),
                  focal_point=(center_lon, center_lat, 5),
                  color='white', intensity=0.2)
plotter.add_light(light3)

# --- GRID Y EJES ---
print("   Añadiendo elementos geográficos...")

# Grid de latitud
for lat in range(int(lat_min), int(lat_max) + 1, 5):
    lon_line = np.linspace(lon_min, lon_max, 100)
    lat_line = np.full(100, lat)
    z_line = np.zeros(100) - 0.2

    points = np.c_[lon_line, lat_line, z_line]
    line = pv.lines_from_points(points)
    plotter.add_mesh(line, color='gray', opacity=0.3, line_width=1)

# Grid de longitud
for lon in range(int(lon_min), int(lon_max) + 1, 10):
    lat_line = np.linspace(lat_min, lat_max, 100)
    lon_line = np.full(100, lon)
    z_line = np.zeros(100) - 0.2

    points = np.c_[lon_line, lat_line, z_line]
    line = pv.lines_from_points(points)
    plotter.add_mesh(line, color='gray', opacity=0.3, line_width=1)

# --- CONFIGURAR CÁMARA (VISTA MÁS HORIZONTAL) ---
print("   Configurando vista horizontal panorámica...")

# Vista más horizontal (elevación baja, mayor distancia)
# Posición de la cámara: más lejos y más baja para perspectiva horizontal
camera_distance = 60  # Distancia de la cámara
camera_elevation = 15  # Altura sobre el centro (más bajo = más horizontal)
camera_azimuth = -45  # Ángulo horizontal (grados)

# Calcular posición de la cámara
azimuth_rad = np.radians(camera_azimuth)
camera_x = center_lon + camera_distance * np.cos(azimuth_rad)
camera_y = center_lat + camera_distance * np.sin(azimuth_rad)
camera_z = camera_elevation

plotter.camera_position = [
    (camera_x, camera_y, camera_z),  # Posición de la cámara
    (center_lon, center_lat, center_z),  # Punto focal
    (0, 0, 1)  # Vector "arriba"
]

# Ajustar campo de visión para vista panorámica
plotter.camera.view_angle = 45  # Ángulo de visión amplio

# --- AÑADIR TÍTULO ---
plotter.add_text(
    f'Estructura Volumétrica 3D: Nubes + Topografía\n'
    f'Región Caribe | ECMWF {fecha} {corrida}Z +{lead_time}h\n'
    f'Creado por: Osmany Lorenzo Amaro',
    position='upper_left',
    font_size=12,
    color='white',
    font='arial'
)

# --- RENDERIZAR Y GUARDAR ---
print("\n💾 Renderizando imagen de alta calidad...")

output_file = f'clouds_terrain_volumetric_3d_{fecha}_{corrida}z_lead{lead_time}h.png'
plotter.screenshot(output_file, transparent_background=False, scale=2)  # scale=2 para mayor calidad

print(f"✅ Imagen guardada: {output_file}")

# Mostrar en notebook si es posible
if EN_COLAB:
    from IPython.display import Image, display
    display(Image(filename=output_file))
    print("\n🖼️  Imagen mostrada en Colab")

plotter.close()

# --- ESTADÍSTICAS FINALES ---
print("\n" + "=" * 70)
print("📊 ESTADÍSTICAS DE LA VISUALIZACIÓN")
print("=" * 70)
print(f"\n🏔️  Topografía:")
print(f"   • Elevación máxima: {elevation.max():.0f} m")
print(f"   • Elevación mínima: {elevation.min():.0f} m")
print(f"   • Rango: {elevation.max() - elevation.min():.0f} m")

print(f"\n☁️  Nubes:")
print(f"   • Altura máxima: {cloud_height.max():.0f} m")
print(f"   • Altura promedio: {cloud_height[cloud_mask].mean():.0f} m")
print(f"   • Cobertura nubosa: {(cloud_mask.sum() / cloud_mask.size * 100):.1f}%")

print(f"\n🎨 Parámetros de visualización:")
print(f"   • z_scale_terrain = {z_scale_terrain}")
print(f"   • z_scale_clouds = {z_scale_clouds}")
print(f"   • z_offset_clouds = {z_offset_clouds}")
print(f"   • resolution = {resolution}°")
print(f"   • Capas de nubes: {num_layers}")

print("\n" + "=" * 70)
print("🎉 VISUALIZACIÓN COMPLETADA EXITOSAMENTE")
print("=" * 70)
