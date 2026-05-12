"""Descarga, carga y combinacion ponderada de modelos meteorologicos.

Modulo extraido y organizado desde el notebook original `modelos-2.ipynb`.
"""

import urllib.request
import xarray as xr
import os
import numpy as np
from ecmwf.opendata import Client
import s3fs
from datetime import datetime, timedelta

class MultiModeloMeteorologico:
    """
    Clase para descargar y cargar datos de múltiples modelos meteorológicos.
    Soporta: GFS, GFS-IA (GraphCast), ECMWF IFS y ECMWF AIFS
    """

    def __init__(self):
        self.fecha = None
        self.corrida = None
        self.acumulado = None
        self.lead_time = None
        self.datos = {}
        self.carpeta_base = "data/raw/modelos"
        os.makedirs(self.carpeta_base, exist_ok=True)

    def configurar(self, fecha=None, corrida=None, acumulado=None):
        """
        Configura los parámetros de descarga

        Args:
            fecha: str en formato YYYYMMDD (ej: '20251016')
            corrida: str hora de corrida '00', '06', '12', '18'
            acumulado: str '12h', '24h', '5d' o rango personalizado 'f024-f036'
        """
        # Si no se proporcionan parámetros, preguntar interactivamente
        if fecha is None:
            fecha = input("Ingrese la fecha (YYYYMMDD, ej: 20251016): ").strip()

        if corrida is None:
            print("\nCorridas disponibles: 00, 06, 12, 18")
            corrida = input("Ingrese la hora de corrida (00/06/12/18): ").strip()

        if acumulado is None:
            print("\nAcumulados disponibles:")
            print("  12h - Acumulado de 12 horas")
            print("  24h - Acumulado de 24 horas")
            print("  5d  - Acumulado de 5 días (120 horas)")
            print("  Personalizado: f024-f036 (acumulado entre hora 24 y 36)")
            acumulado = input("Ingrese el acumulado deseado (12h/24h/5d/f###-f###): ").strip().lower()

        # Validaciones
        if len(fecha) != 8 or not fecha.isdigit():
            raise ValueError("Fecha debe estar en formato YYYYMMDD")

        if corrida not in ['00', '06', '12', '18']:
            raise ValueError("Corrida debe ser 00, 06, 12 o 18")

        self.fecha = fecha
        self.corrida = corrida
        self.acumulado = acumulado

        # Parsear el acumulado
        self.es_rango = False

        if acumulado in ['12h', '24h', '5d']:
            # Acumulados fijos (desde inicio)
            if acumulado == '12h':
                self.lead_time = 12
                self.step_range = "0-12"
                self.hora_inicio = 0
                self.hora_fin = 12
            elif acumulado == '24h':
                self.lead_time = 24
                self.step_range = "0-24"
                self.hora_inicio = 0
                self.hora_fin = 24
            elif acumulado == '5d':
                self.lead_time = 120
                self.step_range = "0-120"
                self.hora_inicio = 0
                self.hora_fin = 120
        else:
            # Rango personalizado: f024-f036
            import re
            match = re.match(r'f(\d+)-f(\d+)', acumulado)
            if not match:
                raise ValueError("Acumulado debe ser 12h, 24h, 5d o formato f###-f### (ej: f024-f036)")

            self.hora_inicio = int(match.group(1))
            self.hora_fin = int(match.group(2))

            if self.hora_inicio >= self.hora_fin:
                raise ValueError("La hora final debe ser mayor que la hora inicial")

            self.es_rango = True
            self.lead_time = self.hora_fin  # Para compatibilidad
            self.step_range = f"{self.hora_inicio}-{self.hora_fin}"

        print(f"\n✅ Configuración:")
        print(f"   Fecha: {self.fecha}")
        print(f"   Corrida: {self.corrida}Z")
        if self.es_rango:
            print(f"   Acumulado: {self.acumulado} (horas {self.hora_inicio}-{self.hora_fin})")
        else:
            print(f"   Acumulado: {self.acumulado} ({self.lead_time}h)")

    def descargar_gfs(self):
        """Descarga datos del modelo GFS"""
        print(f"\n🌍 Descargando GFS...")

        carpeta_gfs = os.path.join(self.carpeta_base, "gfs")
        os.makedirs(carpeta_gfs, exist_ok=True)

        base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{self.fecha}/{self.corrida}/atmos/"

        if self.es_rango:
            # GFS almacena acumulados desde el inicio (0-N horas)
            # Para obtener un intervalo, necesitamos restar dos archivos
            archivo_fin = f"gfs_{self.acumulado}_{self.fecha}_{self.corrida}z_f{self.hora_fin:03d}.grib2"
            archivo_inicio = f"gfs_{self.acumulado}_{self.fecha}_{self.corrida}z_f{self.hora_inicio:03d}.grib2"

            ruta_fin = os.path.join(carpeta_gfs, archivo_fin)
            ruta_inicio = os.path.join(carpeta_gfs, archivo_inicio)

            # Descargar archivo final
            if not os.path.exists(ruta_fin):
                url = base_url + f"gfs.t{self.corrida}z.pgrb2.0p25.f{self.hora_fin:03d}"
                headers = {"User-Agent": "Mozilla/5.0"}
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    with open(ruta_fin, "wb") as f:
                        f.write(response.read())
                print(f"   ✔ Descargado: {archivo_fin}")
            else:
                print(f"   ✔ Ya existe: {archivo_fin}")

            # Cargar archivo final - buscar mensaje con acumulado desde inicio (0-N)
            ds_fin = xr.open_dataset(
                ruta_fin,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {
                        "typeOfLevel": "surface",
                        "stepType": "accum",
                        "stepRange": f"0-{self.hora_fin}"
                    },
                    "indexpath": ""
                }
            )

            # Descargar y cargar archivo inicio
            if self.hora_inicio > 0:
                if not os.path.exists(ruta_inicio):
                    url = base_url + f"gfs.t{self.corrida}z.pgrb2.0p25.f{self.hora_inicio:03d}"
                    headers = {"User-Agent": "Mozilla/5.0"}
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req) as response:
                        with open(ruta_inicio, "wb") as f:
                            f.write(response.read())
                    print(f"   ✔ Descargado: {archivo_inicio}")
                else:
                    print(f"   ✔ Ya existe: {archivo_inicio}")

                ds_inicio = xr.open_dataset(
                    ruta_inicio,
                    engine="cfgrib",
                    backend_kwargs={
                        "filter_by_keys": {
                            "typeOfLevel": "surface",
                            "stepType": "accum",
                            "stepRange": f"0-{self.hora_inicio}"
                        },
                        "indexpath": ""
                    }
                )

                # Hacer la resta para obtener el intervalo
                precip = ds_fin['tp'].values - ds_inicio['tp'].values
                ds = ds_fin
            else:
                precip = ds_fin['tp'].values
                ds = ds_fin

        else:
            # Acumulado fijo (comportamiento original)
            archivo_original = f"gfs.t{self.corrida}z.pgrb2.0p25.f{self.lead_time:03d}"
            archivo = f"gfs_{self.acumulado}_{self.fecha}_{self.corrida}z.grib2"
            url = base_url + archivo_original
            ruta_local = os.path.join(carpeta_gfs, archivo)

            if not os.path.exists(ruta_local):
                headers = {"User-Agent": "Mozilla/5.0"}
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    data_bytes = response.read()
                with open(ruta_local, "wb") as f:
                    f.write(data_bytes)
                print(f"   ✔ Descargado: {archivo}")
            else:
                print(f"   ✔ Ya existe: {archivo}")

            ds = xr.open_dataset(
                ruta_local,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {
                        "typeOfLevel": "surface",
                        "stepType": "accum",
                        "stepRange": self.step_range
                    },
                    "indexpath": ""
                }
            )
            precip = ds['tp'].values

        self.datos['gfs'] = {
            'dataset': ds,
            'precipitacion': precip,
            'max_precip': np.max(precip),
            'lat': ds['latitude'].values,
            'lon': ds['longitude'].values
        }

        print(f"   📊 Precip. máxima: {self.datos['gfs']['max_precip']:.2f} mm")




    def descargar_ecmwf_ifs(self):
        """Descarga datos del modelo ECMWF IFS (físico)"""
        print(f"\n🇪🇺 Descargando ECMWF IFS (Físico)...")

        carpeta_ifs = os.path.join(self.carpeta_base, "ecmwf_ifs")
        os.makedirs(carpeta_ifs, exist_ok=True)

        client = Client(source="ecmwf")

        if self.es_rango:
            archivo_fin = f"ifs_{self.acumulado}_{self.fecha}_{self.corrida}z_f{self.hora_fin:03d}.grib"
            archivo_inicio = f"ifs_{self.acumulado}_{self.fecha}_{self.corrida}z_f{self.hora_inicio:03d}.grib"

            ruta_fin = os.path.join(carpeta_ifs, archivo_fin)
            ruta_inicio = os.path.join(carpeta_ifs, archivo_inicio)

            # Descargar archivo final
            if not os.path.exists(ruta_fin):
                client.retrieve(
                    date=self.fecha,
                    time=int(self.corrida),
                    step=self.hora_fin,
                    param="tp",
                    stream="oper",
                    target=ruta_fin
                )
                print(f"   ✔ Descargado: {archivo_fin}")
            else:
                print(f"   ✔ Ya existe: {archivo_fin}")

            # Descargar archivo inicio
            if self.hora_inicio > 0:
                if not os.path.exists(ruta_inicio):
                    client.retrieve(
                        date=self.fecha,
                        time=int(self.corrida),
                        step=self.hora_inicio,
                        param="tp",
                        stream="oper",
                        target=ruta_inicio
                    )
                    print(f"   ✔ Descargado: {archivo_inicio}")
                else:
                    print(f"   ✔ Ya existe: {archivo_inicio}")

                # Cargar y restar
                ds_fin = xr.open_dataset(ruta_fin, engine="cfgrib",
                                       backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface", "stepType": "accum"}, "indexpath": ""})
                ds_inicio = xr.open_dataset(ruta_inicio, engine="cfgrib",
                                           backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface", "stepType": "accum"}, "indexpath": ""})

                precip_mm = (ds_fin['tp'].values - ds_inicio['tp'].values) * 1000  # metros a mm
                ds = ds_fin
            else:
                ds = xr.open_dataset(ruta_fin, engine="cfgrib",
                                   backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface", "stepType": "accum"}, "indexpath": ""})
                precip_mm = ds['tp'].values * 1000
        else:
            # Comportamiento original
            archivo = f"ifs_{self.acumulado}_{self.fecha}_{self.corrida}z.grib"
            ruta_local = os.path.join(carpeta_ifs, archivo)

            if not os.path.exists(ruta_local):
                client.retrieve(
                    date=self.fecha,
                    time=int(self.corrida),
                    step=self.lead_time,
                    param="tp",
                    stream="oper",
                    target=ruta_local
                )
                print(f"   ✔ Descargado: {archivo}")
            else:
                print(f"   ✔ Ya existe: {archivo}")

            # Cargar datos
            ds = xr.open_dataset(
                ruta_local,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {
                        "typeOfLevel": "surface",
                        "stepType": "accum"
                    },
                    "indexpath": ""
                }
            )

            precip_mm = ds['tp'].values * 1000  # metros a mm

        self.datos['ecmwf_ifs'] = {
            'dataset': ds,
            'precipitacion': precip_mm,
            'max_precip': np.max(precip_mm),
            'lat': ds['latitude'].values,
            'lon': ds['longitude'].values
        }

        print(f"   📊 Precip. máxima: {self.datos['ecmwf_ifs']['max_precip']:.2f} mm")

    def descargar_ecmwf_aifs(self):
        """Descarga datos del modelo ECMWF AIFS (IA)"""
        print(f"\n🤖 Descargando ECMWF AIFS (IA)...")

        carpeta_aifs = os.path.join(self.carpeta_base, "ecmwf_aifs")
        os.makedirs(carpeta_aifs, exist_ok=True)

        client = Client(source="ecmwf", model="aifs-single")

        if self.es_rango:
            archivo_fin = f"aifs_{self.acumulado}_{self.fecha}_{self.corrida}z_f{self.hora_fin:03d}.grib"
            archivo_inicio = f"aifs_{self.acumulado}_{self.fecha}_{self.corrida}z_f{self.hora_inicio:03d}.grib"

            ruta_fin = os.path.join(carpeta_aifs, archivo_fin)
            ruta_inicio = os.path.join(carpeta_aifs, archivo_inicio)

            # Descargar archivo final
            if not os.path.exists(ruta_fin):
                client.retrieve(
                    date=self.fecha,
                    time=int(self.corrida),
                    step=self.hora_fin,
                    type="fc",
                    param="tp",
                    target=ruta_fin
                )
                print(f"   ✔ Descargado: {archivo_fin}")
            else:
                print(f"   ✔ Ya existe: {archivo_fin}")

            # Descargar archivo inicio
            if self.hora_inicio > 0:
                if not os.path.exists(ruta_inicio):
                    client.retrieve(
                        date=self.fecha,
                        time=int(self.corrida),
                        step=self.hora_inicio,
                        type="fc",
                        param="tp",
                        target=ruta_inicio
                    )
                    print(f"   ✔ Descargado: {archivo_inicio}")
                else:
                    print(f"   ✔ Ya existe: {archivo_inicio}")

                # Cargar y restar
                ds_fin = xr.open_dataset(ruta_fin, engine="cfgrib",
                                       backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface", "stepType": "accum"}, "indexpath": ""})
                ds_inicio = xr.open_dataset(ruta_inicio, engine="cfgrib",
                                           backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface", "stepType": "accum"}, "indexpath": ""})

                precip = ds_fin['tp'].values - ds_inicio['tp'].values
                ds = ds_fin
            else:
                ds = xr.open_dataset(ruta_fin, engine="cfgrib",
                                   backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface", "stepType": "accum"}, "indexpath": ""})
                precip = ds['tp'].values
        else:
            # Comportamiento original
            archivo = f"aifs_{self.acumulado}_{self.fecha}_{self.corrida}z.grib"
            ruta_local = os.path.join(carpeta_aifs, archivo)

            if not os.path.exists(ruta_local):
                client.retrieve(
                    date=self.fecha,
                    time=int(self.corrida),
                    step=self.lead_time,
                    type="fc",
                    param="tp",
                    target=ruta_local
                )
                print(f"   ✔ Descargado: {archivo}")
            else:
                print(f"   ✔ Ya existe: {archivo}")

            # Cargar datos
            ds = xr.open_dataset(
                ruta_local,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {
                        "typeOfLevel": "surface",
                        "stepType": "accum"
                    },
                    "indexpath": ""
                }
            )
            precip = ds['tp'].values

        self.datos['ecmwf_aifs'] = {
            'dataset': ds,
            'precipitacion': precip,
            'max_precip': np.max(precip),
            'lat': ds['latitude'].values,
            'lon': ds['longitude'].values
        }

        print(f"   📊 Precip. máxima: {self.datos['ecmwf_aifs']['max_precip']:.2f} mm")

    def descargar_gfs_graphcast(self):
        """Descarga datos del modelo GFS-GraphCast (IA)"""
        print(f"\n🤖 Descargando GFS-GraphCast (IA)...")

        carpeta_graphcast = os.path.join(self.carpeta_base, "gfs_graphcast")
        os.makedirs(carpeta_graphcast, exist_ok=True)

        fs = s3fs.S3FileSystem(anon=True)

        if self.es_rango:
            # Para GraphCast, cada archivo ya contiene el acumulado del período específico
            archivo = f"graphcast_{self.acumulado}_{self.fecha}_{self.corrida}z.grib2"
            ruta_local = os.path.join(carpeta_graphcast, archivo)

            if not os.path.exists(ruta_local):
                file_path = f'noaa-nws-graphcastgfs-pds/graphcastgfs.{self.fecha}/{self.corrida}/forecasts_13_levels/graphcastgfs.t{self.corrida}z.pgrb2.0p25.f{self.hora_fin:03d}'
                with fs.open(file_path, 'rb') as fsrc:
                    with open(ruta_local, 'wb') as fdst:
                        fdst.write(fsrc.read())
                print(f"   ✔ Descargado: {archivo}")
            else:
                print(f"   ✔ Ya existe: {archivo}")

            # Cargar datos
            ds = xr.open_dataset(
                ruta_local,
                engine='cfgrib',
                backend_kwargs={
                    "filter_by_keys": {
                        "typeOfLevel": "surface",
                        "stepType": "accum",
                        "shortName": "tp"
                    },
                    "indexpath": ""
                }
            )
            precip = ds['tp'].values

        else:
            # Comportamiento original
            archivo = f"graphcast_{self.acumulado}_{self.fecha}_{self.corrida}z.grib2"
            ruta_local = os.path.join(carpeta_graphcast, archivo)

            if not os.path.exists(ruta_local):
                file_path = f'noaa-nws-graphcastgfs-pds/graphcastgfs.{self.fecha}/{self.corrida}/forecasts_13_levels/graphcastgfs.t{self.corrida}z.pgrb2.0p25.f{self.lead_time:03d}'

                with fs.open(file_path, 'rb') as fsrc:
                    with open(ruta_local, 'wb') as fdst:
                        fdst.write(fsrc.read())
                print(f"   ✔ Descargado: {archivo}")
            else:
                print(f"   ✔ Ya existe: {archivo}")

            # Cargar datos
            ds = xr.open_dataset(
                ruta_local,
                engine='cfgrib',
                backend_kwargs={
                    "filter_by_keys": {
                        "typeOfLevel": "surface",
                        "stepType": "accum",
                        "shortName": "tp"
                    },
                    "indexpath": ""
                }
            )
            precip = ds['tp'].values

        self.datos['gfs_graphcast'] = {
            'dataset': ds,
            'precipitacion': precip,
            'max_precip': np.max(precip),
            'lat': ds['latitude'].values,
            'lon': ds['longitude'].values
        }

        print(f"   📊 Precip. máxima: {self.datos['gfs_graphcast']['max_precip']:.2f} mm")

    def cargar_todos_los_modelos(self):
        """Descarga y carga todos los modelos configurados"""
        if self.fecha is None or self.corrida is None or self.acumulado is None:
            raise ValueError("Debe configurar la clase primero usando configurar()")

        print("\n" + "="*60)
        print("🚀 INICIANDO DESCARGA DE TODOS LOS MODELOS")
        print("="*60)

        try:
            self.descargar_gfs()
        except Exception as e:
            print(f"   ❌ Error en GFS: {e}")

        try:
            self.descargar_ecmwf_ifs()
        except Exception as e:
            print(f"   ❌ Error en ECMWF IFS: {e}")

        try:
            self.descargar_ecmwf_aifs()
        except Exception as e:
            print(f"   ❌ Error en ECMWF AIFS: {e}")

        try:
            self.descargar_gfs_graphcast()
        except Exception as e:
            print(f"   ❌ Error en GFS-GraphCast: {e}")

        print("\n" + "="*60)
        print("✅ DESCARGA COMPLETADA")
        print("="*60)
        self.mostrar_resumen()

    def mostrar_resumen(self):
        """Muestra un resumen de los datos cargados"""
        print(f"\n📋 RESUMEN DE DATOS CARGADOS:")
        print(f"   Fecha: {self.fecha}")
        print(f"   Corrida: {self.corrida}Z")
        print(f"   Acumulado: {self.acumulado}")
        print(f"\n   Modelos cargados: {len(self.datos)}")

        for modelo, datos in self.datos.items():
            print(f"\n   🔹 {modelo.upper()}:")
            print(f"      Max precip: {datos['max_precip']:.2f} mm")
            print(f"      Shape: {datos['precipitacion'].shape}")

    def obtener_precipitacion(self, modelo):
        """
        Obtiene la precipitación de un modelo específico

        Args:
            modelo: str - 'gfs', 'ecmwf_ifs', 'ecmwf_aifs', 'gfs_graphcast'

        Returns:
            numpy array con valores de precipitación en mm
        """
        if modelo not in self.datos:
            raise ValueError(f"Modelo '{modelo}' no cargado. Modelos disponibles: {list(self.datos.keys())}")

        return self.datos[modelo]['precipitacion']

    def obtener_coordenadas(self, modelo):
        """Obtiene las coordenadas (lat, lon) de un modelo"""
        if modelo not in self.datos:
            raise ValueError(f"Modelo '{modelo}' no cargado")

        return self.datos[modelo]['lat'], self.datos[modelo]['lon']

    def obtener_dataset(self, modelo):
        """Obtiene el dataset completo de xarray de un modelo"""
        if modelo not in self.datos:
            raise ValueError(f"Modelo '{modelo}' no cargado")

        return self.datos[modelo]['dataset']

    def _interpolar_a_rejilla_comun(self, modelo_key, lon_common, lat_common):
        """
        Interpola los datos de precipitación de un modelo a la rejilla común.
        Usa directamente los datos ya procesados en self.datos
        """
        from scipy.interpolate import RegularGridInterpolator

        # Obtener datos ya procesados (en mm)
        precip_orig_mm = self.datos[modelo_key]['precipitacion']
        lat_orig = self.datos[modelo_key]['lat']
        lon_orig = self.datos[modelo_key]['lon']

        # Extraer datos si tienen dimensión temporal
        if precip_orig_mm.ndim == 3:
            precip_orig_mm = precip_orig_mm[0, :, :]

        # Convertir longitudes 0-360 a -180-180 si es necesario
        if lon_orig.min() >= 0 and lon_orig.max() > 180:
            lon_orig_converted = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
        else:
            lon_orig_converted = lon_orig

        # Ordenar longitudes si es necesario
        if lon_orig_converted[0] > lon_orig_converted[-1]:
            idx_sort = np.argsort(lon_orig_converted)
            lon_orig_sorted = lon_orig_converted[idx_sort]
            precip_orig_mm = precip_orig_mm[:, idx_sort]
        else:
            lon_orig_sorted = lon_orig_converted

        # Verificar orden de latitudes
        if lat_orig[0] > lat_orig[-1]:
            lat_orig_sorted = lat_orig[::-1]
            precip_orig_mm = precip_orig_mm[::-1, :]
        else:
            lat_orig_sorted = lat_orig

        # Crear interpolador
        interpolador = RegularGridInterpolator(
            (lat_orig_sorted, lon_orig_sorted),
            precip_orig_mm,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Crear puntos de la rejilla común
        lat_grid_common, lon_grid_common = np.meshgrid(lat_common, lon_common, indexing='ij')
        points = np.column_stack([lat_grid_common.ravel(), lon_grid_common.ravel()])

        # Interpolar
        precip_interp = interpolador(points).reshape(len(lat_common), len(lon_common))

        # Asegurar que no hay valores negativos
        precip_interp = np.maximum(precip_interp, 0.0)

        return precip_interp

    def crear_multimodelo(self, pesos=None, factores_calibracion=None, resolucion=0.25):
        """
        Crea un multimodelo interpolando todos los modelos a una rejilla común.

        Args:
            pesos: dict con pesos para cada modelo.
                   Ej: {'gfs': 0.40, 'ecmwf_ifs': 0.35, 'ecmwf_aifs': 0.10, 'gfs_graphcast': 0.15}
            factores_calibracion: dict con factores de calibración para cada modelo.
                   Ej: {'gfs': 1.5,
                   'ecmwf_ifs': 1.5, 'ecmwf_aifs': 2.0, 'gfs_graphcast': 2.5}
            resolucion: float, resolución de la rejilla común en grados (default: 0.25)

        Returns:
            xr.DataArray con el multimodelo en rejilla común
        """
        if len(self.datos) == 0:
            raise ValueError("No hay modelos cargados. Ejecute cargar_todos_los_modelos() primero.")

        print(f"\n🔄 Creando multimodelo con rejilla común ({resolucion}°)...")

        # Definir pesos por defecto
        if pesos is None:
            pesos = {
                'gfs': 0.35,
                'ecmwf_ifs': 0.40,
                'ecmwf_aifs': 0.15,
                'gfs_graphcast': 0.10
            }

        # Definir factores de calibración por defecto
        if factores_calibracion is None:
            factores_calibracion = {
                'gfs': 1.6,
                'ecmwf_ifs': 1.6,
                'ecmwf_aifs': 2.5,
                'gfs_graphcast': 3
            }

        # Validar que la suma de pesos es 1.0
        suma_pesos = sum(pesos.get(modelo, 0) for modelo in self.datos.keys())
        if abs(suma_pesos - 1.0) > 0.001:
            print(f"⚠️  Advertencia: La suma de pesos es {suma_pesos:.3f}, se normalizará a 1.0")
            factor_norm = 1.0 / suma_pesos
            pesos = {k: v * factor_norm for k, v in pesos.items()}

        # Definir rejilla común
        lon_common = np.arange(-180, 180, resolucion)
        lat_common = np.arange(-90, 90 + resolucion, resolucion)

        print(f"   Rejilla: {len(lat_common)} × {len(lon_common)} puntos")

        # Interpolar cada modelo
        modelos_interpolados = {}

        for modelo_key in self.datos.keys():
            print(f"   📍 Interpolando {modelo_key}...")

            # Interpolar usando los datos ya procesados en mm
            precip_interp = self._interpolar_a_rejilla_comun(
                modelo_key, lon_common, lat_common
            )

            print(f"      Original - Min: {self.datos[modelo_key]['precipitacion'].min():.2f}, Max: {self.datos[modelo_key]['precipitacion'].max():.2f} mm")
            print(f"      Interpolado - Min: {precip_interp.min():.2f}, Max: {precip_interp.max():.2f} mm")

            # Aplicar factor de calibración
            factor = factores_calibracion.get(modelo_key, 1.0)
            precip_interp = precip_interp * factor

            modelos_interpolados[modelo_key] = precip_interp

        # Verificar que todos tienen el mismo shape
        shapes = [arr.shape for arr in modelos_interpolados.values()]
        if len(set(shapes)) != 1:
            raise ValueError("ERROR: Los modelos interpolados tienen diferentes shapes")

        print(f"   ✓ Todos los modelos interpolados: {shapes[0]}")

        # Crear multimodelo ponderado
        print(f"   🔢 Calculando promedio ponderado...")
        precip_multimodelo = np.zeros_like(list(modelos_interpolados.values())[0])

        for modelo_key, precip_interp in modelos_interpolados.items():
            peso = pesos.get(modelo_key, 0)
            precip_multimodelo += peso * precip_interp
            print(f"      {modelo_key}: peso={peso:.2f}, factor={factores_calibracion.get(modelo_key, 1.0):.1f}")

        # Extraer tiempo del primer modelo disponible
        primer_modelo = list(self.datos.values())[0]['dataset']
        if hasattr(primer_modelo, 'time'):
            if np.isscalar(primer_modelo.time.values) or primer_modelo.time.values.ndim == 0:
                time_value = [primer_modelo.time.values.item()]
            else:
                time_value = [primer_modelo.time.values[0]]
        else:
            time_value = [np.datetime64('2024-01-01')]

        # Crear DataArray
        ds_multimodelo = xr.DataArray(
            precip_multimodelo[np.newaxis, :, :],
            coords={
                'time': time_value,
                'latitude': lat_common,
                'longitude': lon_common
            },
            dims=['time', 'latitude', 'longitude'],
            attrs={
                'long_name': f'Precipitación acumulada {self.acumulado} - Multimodelo',
                'units': 'mm',
                'description': 'Promedio ponderado con rejilla común',
                'grid': f'Common {resolucion}° × {resolucion}° grid, -180 to 180 longitude',
                'pesos': str(pesos),
                'factores_calibracion': str(factores_calibracion)
            }
        )

        # Guardar en el objeto
        self.multimodelo = ds_multimodelo
        self.modelos_interpolados = modelos_interpolados
        self.lon_common = lon_common
        self.lat_common = lat_common

        print(f"\n   ✅ Multimodelo creado exitosamente")
        print(f"      Min: {precip_multimodelo.min():.2f} mm")
        print(f"      Max: {precip_multimodelo.max():.2f} mm")
        print(f"      Mean: {precip_multimodelo.mean():.2f} mm")

        return ds_multimodelo

    def comparar_punto(self, lat, lon):
        """
        Compara los valores de precipitación de todos los modelos en un punto específico.

        Args:
            lat: float, latitud del punto
            lon: float, longitud del punto
        """
        if not hasattr(self, 'multimodelo'):
            raise ValueError("Debe crear el multimodelo primero usando crear_multimodelo()")

        # Encontrar índice más cercano
        idx_lat = np.argmin(np.abs(self.lat_common - lat))
        idx_lon = np.argmin(np.abs(self.lon_common - lon))

        print(f"\n📍 Comparación en punto ({lat:.2f}°N, {lon:.2f}°E)")
        print(f"   Coordenadas exactas: ({self.lat_common[idx_lat]:.2f}°N, {self.lon_common[idx_lon]:.2f}°E)")
        print(f"\n   Precipitación por modelo:")

        for modelo_key, precip_interp in self.modelos_interpolados.items():
            valor = precip_interp[idx_lat, idx_lon]
            print(f"      {modelo_key:20s}: {valor:6.2f} mm")

        valor_multi = self.multimodelo.values[0, idx_lat, idx_lon]
        print(f"      {'MULTIMODELO':20s}: {valor_multi:6.2f} mm")

        return valor_multi