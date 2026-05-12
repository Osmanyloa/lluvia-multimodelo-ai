import os
import numpy as np
import xarray as xr
import cfgrib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ecmwf.opendata import Client

# --- CONFIGURACIÓN ---
# IMPORTANTE: Para pronósticos futuros, ajusta estos parámetros
fecha = "20251027"  # YYYYMMDD - Fecha de la corrida del modelo
corrida = "00"  # Hora de corrida: 00 o 12
lead_time = 95  # Horas de pronóstico desde la corrida (0-240)

carpeta = "./datos_ecmwf"
os.makedirs(carpeta, exist_ok=True)

# --- LÍMITES DEL CARIBE ---
lon_min, lon_max = -95, -55
lat_min, lat_max = 5, 30

# --- NIVELES DE PRESIÓN A VISUALIZAR ---
niveles_presion = [850, 700, 500, 400, 200]  # hPa

# --- PARÁMETROS PARA VECTORES DE VIENTO ---
vector_density = 8  # Mostrar 1 de cada N puntos (Mayor = más espaciado)
# Ajusta estos valores si las flechas son muy grandes/pequeñas

# --- FUNCIÓN AUXILIAR PARA CALCULAR FECHA VÁLIDA ---
def calcular_fecha_valida(fecha_str, hora_str, lead_hours):
    """Calcula la fecha válida del pronóstico"""
    from datetime import datetime, timedelta

    year = int(fecha_str[:4])
    month = int(fecha_str[4:6])
    day = int(fecha_str[6:8])
    hour = int(hora_str)

    dt_corrida = datetime(year, month, day, hour)
    dt_valido = dt_corrida + timedelta(hours=lead_hours)

    return dt_valido.strftime("%Y-%m-%d %H:00 UTC")

print("=" * 70)
print("DESCARGANDO DATOS DE GEOPOTENCIAL Y VIENTO EN MÚLTIPLES NIVELES")
print("=" * 70)
print(f"📅 Configuración:")
print(f"   Fecha de corrida: {fecha}")
print(f"   Hora de corrida: {corrida}Z")
print(f"   Lead time: {lead_time} horas")
print(f"   Pronóstico válido para: {calcular_fecha_valida(fecha, corrida, lead_time)}")
print(f"   Niveles: {niveles_presion}")

archivo = f"ifs_pl_multi_wind_{fecha}_{corrida}z_lead{lead_time}h.grib"
ruta_local = os.path.join(carpeta, archivo)

if not os.path.exists(ruta_local):
    print(f"\n📥 Descargando pronóstico de geopotencial y viento...")

    try:
        client = Client(source="ecmwf")

        print(f"   Intentando descargar desde ECMWF...")

        # Descargar geopotencial (gh), componente U (u) y componente V (v) del viento
        client.retrieve(
            date=int(fecha),
            time=int(corrida),
            step=lead_time,
            stream="oper",
            type="fc",
            levtype="pl",
            param=["gh", "u", "v"],  # Geopotencial y componentes del viento
            levelist=niveles_presion,
            target=ruta_local
        )
        print(f"✅ Descarga completada: {ruta_local}")

    except Exception as e:
        print(f"❌ Error en descarga: {e}")
        print("\n" + "=" * 70)
        print("💡 GUÍA PARA OBTENER DATOS")
        print("=" * 70)
        print("\n📌 OPCIONES PARA DIFERENTES ESCENARIOS:")
        print("\n1️⃣  PRONÓSTICOS RECIENTES (últimos 2-3 días):")
        print("   - Usa la fecha de HOY o AYER como 'fecha'")
        print("   - Ajusta 'lead_time' para proyectar al futuro")

        print("\n💡 SUGERENCIA ACTUAL:")
        from datetime import datetime, timedelta
        hoy = datetime.now()
        ayer = hoy - timedelta(days=1)
        print(f"   - Usa fecha: {ayer.strftime('%Y%m%d')}")
        print(f"   - Con lead_time apropiado para tu fecha objetivo")
        print("=" * 70)

        exit(1)
else:
    print(f"✅ Archivo existente encontrado: {ruta_local}")

# --- CARGAR DATOS ---
print("\n" + "=" * 70)
print("CARGANDO GEOPOTENCIAL Y VIENTO EN TODOS LOS NIVELES")
print("=" * 70)

try:
    # Cargar todos los datasets
    datasets = cfgrib.open_datasets(ruta_local)
    print(f"✅ Número de datasets: {len(datasets)}")

    # Encontrar datasets de geopotencial y viento
    ds_gh = None
    ds_u = None
    ds_v = None

    for ds in datasets:
        if 'gh' in ds.data_vars:
            ds_gh = ds
            print(f"   ✓ Geopotencial encontrado")
        if 'u' in ds.data_vars:
            ds_u = ds
            print(f"   ✓ Componente U del viento encontrado")
        if 'v' in ds.data_vars:
            ds_v = ds
            print(f"   ✓ Componente V del viento encontrado")

    if ds_gh is None or ds_u is None or ds_v is None:
        print("❌ No se encontraron todas las variables necesarias")
        exit(1)

    if 'isobaricInhPa' in ds_gh.coords:
        niveles_disponibles = ds_gh.coords['isobaricInhPa'].values
        print(f"   Niveles disponibles: {niveles_disponibles} hPa")

except Exception as e:
    print(f"❌ Error cargando dataset: {e}")
    exit(1)

# --- EXTRAER DATOS POR NIVEL ---
print("\n" + "=" * 70)
print("PROCESANDO DATOS POR NIVEL")
print("=" * 70)

datos_niveles = {}

for nivel in niveles_presion:
    try:
        # Seleccionar nivel específico
        gh_nivel = ds_gh.sel(isobaricInhPa=nivel)
        u_nivel = ds_u.sel(isobaricInhPa=nivel)
        v_nivel = ds_v.sel(isobaricInhPa=nivel)

        z_raw = gh_nivel['gh']
        u_raw = u_nivel['u']
        v_raw = v_nivel['v']

        print(f"\n📊 Nivel {nivel} hPa:")
        print(f"   Geopotencial - Rango: [{z_raw.min().values:.0f}, {z_raw.max().values:.0f}] mgp")
        print(f"   Viento U - Rango: [{u_raw.min().values:.1f}, {u_raw.max().values:.1f}] m/s")
        print(f"   Viento V - Rango: [{v_raw.min().values:.1f}, {v_raw.max().values:.1f}] m/s")

        # Extraer coordenadas
        lons_full = gh_nivel['longitude'].values
        lats_full = gh_nivel['latitude'].values

        # Recortar al Caribe
        lon_mask = (lons_full >= lon_min) & (lons_full <= lon_max)
        lat_mask = (lats_full >= lat_min) & (lats_full <= lat_max)

        lons = lons_full[lon_mask]
        lats = lats_full[lat_mask]
        z_caribe = z_raw.values[np.ix_(lat_mask, lon_mask)]
        u_caribe = u_raw.values[np.ix_(lat_mask, lon_mask)]
        v_caribe = v_raw.values[np.ix_(lat_mask, lon_mask)]

        # Calcular magnitud del viento
        wind_speed = np.sqrt(u_caribe**2 + v_caribe**2)

        datos_niveles[nivel] = {
            'lons': lons,
            'lats': lats,
            'z': z_caribe,
            'u': u_caribe,
            'v': v_caribe,
            'speed': wind_speed,
            'z_min': z_caribe.min(),
            'z_max': z_caribe.max(),
            'z_media': z_caribe.mean(),
            'wind_max': wind_speed.max(),
            'wind_media': wind_speed.mean()
        }

        print(f"   Velocidad viento - Máx: {wind_speed.max():.1f} m/s, Media: {wind_speed.mean():.1f} m/s")

    except Exception as e:
        print(f"   ❌ Error procesando nivel {nivel} hPa: {e}")

if len(datos_niveles) == 0:
    print("\n❌ No se pudieron cargar datos")
    exit(1)

print(f"\n✅ {len(datos_niveles)} niveles procesados correctamente")

# --- ASIGNAR POSICIONES EN EJE Z ---
posiciones_z = {
    850: 10,
    700: 25,
    500: 40,
    400: 55,
    250: 70,
    200: 82,
    100: 95
}

# --- VISUALIZACIÓN 3D CON MÚLTIPLES CAPAS, MAPA BASE Y VECTORES ---
print("\n" + "=" * 70)
print("GENERANDO VISUALIZACIÓN 3D MULTI-NIVEL CON VECTORES DE VIENTO")
print("=" * 70)

fig = plt.figure(figsize=(24, 20))
ax = fig.add_subplot(111, projection='3d')

# Dibujar mapa base en Z=0
print("   Proyectando mapa base en Z=0...")

from cartopy.feature import COASTLINE, BORDERS

# Dibujar líneas de costa
for geom in COASTLINE.geometries():
    if geom.geom_type == 'LineString':
        coords = np.array(geom.coords)
        lons_costa = coords[:, 0]
        lats_costa = coords[:, 1]

        mask = ((lons_costa >= lon_min - 5) & (lons_costa <= lon_max + 5) &
                (lats_costa >= lat_min - 5) & (lats_costa <= lat_max + 5))

        if np.any(mask):
            lons_segmento = lons_costa[mask]
            lats_segmento = lats_costa[mask]
            z_segmento = np.zeros_like(lons_segmento) + 0.1

            ax.plot(lons_segmento, lats_segmento, z_segmento,
                   color='black', linewidth=1.5, alpha=0.8, zorder=10)
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            coords = np.array(line.coords)
            lons_costa = coords[:, 0]
            lats_costa = coords[:, 1]

            mask = ((lons_costa >= lon_min - 5) & (lons_costa <= lon_max + 5) &
                    (lats_costa >= lat_min - 5) & (lats_costa <= lat_max + 5))

            if np.any(mask):
                lons_segmento = lons_costa[mask]
                lats_segmento = lats_costa[mask]
                z_segmento = np.zeros_like(lons_segmento) + 0.1

                ax.plot(lons_segmento, lats_segmento, z_segmento,
                       color='black', linewidth=1.5, alpha=0.8, zorder=10)

# Dibujar fronteras
for geom in BORDERS.geometries():
    if geom.geom_type == 'LineString':
        coords = np.array(geom.coords)
        lons_frontera = coords[:, 0]
        lats_frontera = coords[:, 1]

        mask = ((lons_frontera >= lon_min - 5) & (lons_frontera <= lon_max + 5) &
                (lats_frontera >= lat_min - 5) & (lats_frontera <= lat_max + 5))

        if np.any(mask):
            lons_segmento = lons_frontera[mask]
            lats_segmento = lats_frontera[mask]
            z_segmento = np.zeros_like(lons_segmento) + 0.1

            ax.plot(lons_segmento, lats_segmento, z_segmento,
                   color='gray', linewidth=0.8, alpha=0.6, linestyle='--', zorder=9)
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            coords = np.array(line.coords)
            lons_frontera = coords[:, 0]
            lats_frontera = coords[:, 1]

            mask = ((lons_frontera >= lon_min - 5) & (lons_frontera <= lon_max + 5) &
                    (lats_frontera >= lat_min - 5) & (lats_frontera <= lat_max + 5))

            if np.any(mask):
                lons_segmento = lons_frontera[mask]
                lats_segmento = lats_frontera[mask]
                z_segmento = np.zeros_like(lons_segmento) + 0.1

                ax.plot(lons_segmento, lats_segmento, z_segmento,
                       color='gray', linewidth=0.8, alpha=0.6, linestyle='--', zorder=9)

# Grid de lat/lon en Z=0
for lon in range(int(lon_min), int(lon_max) + 1, 10):
    lat_line = np.linspace(lat_min, lat_max, 50)
    lon_line = np.full_like(lat_line, lon)
    z_line = np.zeros_like(lat_line) + 0.05
    ax.plot(lon_line, lat_line, z_line, color='gray', linewidth=0.5, alpha=0.3, zorder=2)

for lat in range(int(lat_min), int(lat_max) + 1, 5):
    lon_line = np.linspace(lon_min, lon_max, 50)
    lat_line = np.full_like(lon_line, lat)
    z_line = np.zeros_like(lat_line) + 0.05
    ax.plot(lon_line, lat_line, z_line, color='gray', linewidth=0.5, alpha=0.3, zorder=2)

print("   ✓ Mapa base proyectado")

# Paleta de colores
cmap_unico = 'RdYlBu_r'
factor_escala = 0.08

print("   Generando capas de geopotencial con vectores de viento...")

# Plotear cada nivel
for i, nivel in enumerate(sorted(niveles_presion, reverse=True)):
    if nivel not in datos_niveles:
        continue

    datos = datos_niveles[nivel]
    lons = datos['lons']
    lats = datos['lats']
    z_geo = datos['z']
    u = datos['u']
    v = datos['v']

    # Crear malla
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Posición base en Z para este nivel
    z_base = posiciones_z[nivel]

    # Agregar ondulaciones escaladas
    z_desviacion = z_geo - z_geo.mean()
    z_superficie = z_base + (z_desviacion * factor_escala)

    # Normalizar colores
    norm = plt.Normalize(vmin=z_geo.min(), vmax=z_geo.max())
    cmap = cm.get_cmap(cmap_unico)
    colors = cmap(norm(z_geo))

    # Plotear superficie
    surf = ax.plot_surface(
        lon2d, lat2d, z_superficie,
        facecolors=colors,
        alpha=0.88,  # Restaurado al valor original
        edgecolor='none',
        antialiased=True,
        shade=True,
        lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
        zorder=100 + i
    )

    # Contornos sobre la superficie (más oscuros para contraste)
    contours = ax.contour(
        lon2d, lat2d, z_superficie,
        levels=12,
        colors='black',
        linewidths=0.8,
        alpha=0.7,  # Restaurado al valor original
        linestyles='solid',
        zorder=200 + i
    )

    # VECTORES DE VIENTO
    # Submuestrear los datos para evitar saturación
    lon2d_sub = lon2d[::vector_density, ::vector_density]
    lat2d_sub = lat2d[::vector_density, ::vector_density]
    z_sub = z_superficie[::vector_density, ::vector_density]
    u_sub = u[::vector_density, ::vector_density]
    v_sub = v[::vector_density, ::vector_density]

    # Aplanar arrays para usar con quiver
    lon_flat = lon2d_sub.flatten()
    lat_flat = lat2d_sub.flatten()
    z_flat = z_sub.flatten()
    u_flat = u_sub.flatten()
    v_flat = v_sub.flatten()

    # Calcular la magnitud del viento para colorear Y escalar
    wind_mag = np.sqrt(u_flat**2 + v_flat**2)

    # Normalizar para obtener colores (rojo = rápido, azul = lento)
    norm_wind = plt.Normalize(vmin=wind_mag.min(), vmax=wind_mag.max())
    wind_colors = plt.cm.jet(norm_wind(wind_mag))

    # ESCALAR las componentes U y V por la magnitud
    # Esto hace que vectores con más velocidad sean más largos
    # NO usar normalize=True para que respeten sus magnitudes reales
    scale_factor = 0.08  # Factor de escala global reducido para flechas más pequeñas
    u_scaled = u_flat * scale_factor
    v_scaled = v_flat * scale_factor

    # Dibujar TODOS los vectores de una vez con quiver
    # SIN normalize=True para que el tamaño refleje la intensidad
    quiv = ax.quiver(lon_flat, lat_flat, z_flat,
                     u_scaled, v_scaled, np.zeros_like(u_flat),  # dz = 0 (horizontal)
                     colors=wind_colors,
                     arrow_length_ratio=0.3,  # Reducido para flechas variables
                     linewidth=2.0,
                     alpha=0.95,
                     zorder=300 + i)

    print(f"   ✓ Nivel {nivel} hPa procesado (superficie + {len(lon_flat)} vectores)")

print("   ✓ Todas las capas generadas")

# Líneas de referencia
for lon_corner in [lon_min, lon_max]:
    for lat_corner in [lat_min, lat_max]:
        z_linea = [posiciones_z[n] for n in sorted(niveles_presion)]
        ax.plot([lon_corner]*len(z_linea), [lat_corner]*len(z_linea), z_linea,
               'gray', linestyle=':', linewidth=0.8, alpha=0.4)

for nivel in niveles_presion:
    z_nivel = posiciones_z[nivel]
    corners_lon = [lon_min, lon_max, lon_max, lon_min, lon_min]
    corners_lat = [lat_min, lat_min, lat_max, lat_max, lat_min]
    corners_z = [z_nivel] * 5
    ax.plot(corners_lon, corners_lat, corners_z,
           'gray', linestyle='--', linewidth=0.5, alpha=0.3)

# Configuración de ejes
ax.set_xlabel('Longitud (°)', fontsize=13, labelpad=12)
ax.set_ylabel('Latitud (°)', fontsize=13, labelpad=12)
ax.set_zlabel('Nivel de Presión (hPa)', fontsize=13, labelpad=12)

z_ticks = [posiciones_z[n] for n in sorted(niveles_presion)]
z_labels = [f'{n}' for n in sorted(niveles_presion)]
ax.set_zticks(z_ticks)
ax.set_zticklabels(z_labels)

# Título
tiempo = ds_gh.get('time', ds_gh.get('valid_time', fecha))
fecha_valida = calcular_fecha_valida(fecha, corrida, lead_time)

ax.set_title(
    f'Altura Geopotencial y Vectores de Viento - Región Caribe\n' +
    f'Modelo ECMWF | Corrida: {tiempo.values} | Válido: {fecha_valida}\n' +
    f'Niveles: {", ".join([str(n) for n in sorted(niveles_presion, reverse=True)])} hPa | Lead time: +{lead_time}h',
    fontsize=16,
    fontweight='bold',
    pad=40
)

# Vista óptima
ax.view_init(elev=20, azim=50)
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_zlim(0, 100)

ax.set_box_aspect([
    (lon_max - lon_min),
    (lat_max - lat_min),
    50
])

# Crédito
ax.text2D(0.02, 0.02, 'Creado por: Osmany Lorenzo Amaro',
          transform=ax.transAxes,
          fontsize=12,
          fontweight='bold',
          color='white',
          bbox=dict(boxstyle='round,pad=0.7', facecolor='darkblue', alpha=0.9,
                    edgecolor='white', linewidth=2),
          zorder=1000)

plt.tight_layout()
archivo_3d = f'z_wind_multinivel_caribe_3d_{fecha}_{corrida}z_lead{lead_time}h.png'
plt.savefig(archivo_3d, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Gráfico 3D multi-nivel con vectores guardado: {archivo_3d}")
plt.show()

# --- ESTADÍSTICAS FINALES ---
print("\n" + "=" * 70)
print("📊 ESTADÍSTICAS POR NIVEL")
print("=" * 70)

for nivel in sorted(niveles_presion, reverse=True):
    if nivel in datos_niveles:
        datos = datos_niveles[nivel]
        print(f"\n{nivel} hPa:")
        print(f"  Geopotencial - Media: {datos['z_media']:.0f} mgp")
        print(f"  Geopotencial - Rango: {datos['z_min']:.0f} - {datos['z_max']:.0f} mgp")
        print(f"  Viento - Máx: {datos['wind_max']:.1f} m/s")
        print(f"  Viento - Media: {datos['wind_media']:.1f} m/s")

print("\n" + "=" * 70)
print("🎉 PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)