# Descripcion tecnica

## Problema

Pronosticar acumulados de lluvia exige combinar modelos con fortalezas distintas. Los modelos fisicos representan la dinamica atmosferica mediante ecuaciones, mientras que los modelos de IA pueden capturar patrones aprendidos a partir de grandes volumenes de datos.

## Flujo de trabajo

1. Configurar fecha, corrida y ventana de acumulado.
2. Descargar o localizar archivos GRIB por modelo.
3. Cargar precipitacion acumulada desde cada fuente.
4. Interpolar todos los campos a una rejilla comun.
5. Aplicar factores de calibracion.
6. Calcular el ensamble ponderado.
7. Generar mapas regionales y comparaciones puntuales.

## Modelos incluidos

- GFS: modelo fisico global de NOAA.
- ECMWF IFS: modelo fisico global de ECMWF.
- GraphCast/GFS-IA: prediccion apoyada en IA disponible en buckets publicos.
- ECMWF AIFS: sistema de prediccion con IA de ECMWF.

## Salidas esperadas

- Dataset `xarray` con precipitacion multimodelo.
- Mapas de acumulados de lluvia por region.
- Comparacion puntual entre modelos para validacion rapida.
