# Lluvia Multimodelo AI

Proyecto de pronostico de precipitacion acumulada basado en un ensamble ponderado entre modelos meteorologicos fisicos y modelos con IA.

El objetivo es combinar fuentes como GFS, ECMWF IFS, GraphCast/GFS-IA y ECMWF AIFS en una rejilla comun para producir acumulados de lluvia y mapas regionales listos para analisis meteorologico.

## Que incluye

- Descarga y lectura de salidas GRIB de modelos meteorologicos.
- Configuracion de corrida, fecha y periodos de acumulado: `12h`, `24h`, `5d` o rangos tipo `f024-f144`.
- Interpolacion a una rejilla comun.
- Ensamble ponderado entre modelos fisicos e IA.
- Factores de calibracion por modelo.
- Mapas regionales para Caribe, Cuba, Centroamerica, La Espanola, Antillas Menores, Puerto Rico, Colombia/Venezuela, Estados Unidos, Florida, Texas, Mexico, Iberia y Canarias.

## Estructura

```text
lluvia-multimodelo-ai/
├── src/lluvia_multimodelo/
│   ├── core.py          # Clase principal MultiModeloMeteorologico
│   ├── maps.py          # Funciones de mapas regionales
│   └── geography.py     # Carga de shapefiles y capas geograficas
├── examples/
│   └── run_forecast.py  # Ejemplo reproducible de uso
├── notebooks/
│   └── modelos-2.ipynb  # Notebook original como referencia
├── data/                # Datos locales ignorados por git
├── outputs/             # Mapas y figuras generadas
├── docs/
│   └── project_overview.md
├── requirements.txt
└── pyproject.toml
```

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> Nota: `cfgrib` requiere la libreria del sistema `eccodes`. En Linux suele instalarse con `apt-get install libeccodes-dev`; en macOS puede instalarse con `brew install eccodes`.

## Uso rapido

```python
from lluvia_multimodelo import MultiModeloMeteorologico

modelo = MultiModeloMeteorologico()
modelo.configurar(fecha="20260511", corrida="06", acumulado="f024-f144")
modelo.cargar_todos_los_modelos()

ds_multi = modelo.crear_multimodelo(
    pesos={"gfs": 0.25, "graphcast": 0.25, "ifs": 0.25, "aifs": 0.25}
)

modelo.comparar_punto(lat=40.0, lon=-100.0)
```

## Idea del modelo

El multimodelo combina predictores fisicos tradicionales y predictores apoyados en IA. La filosofia inicial del proyecto es tratar ambos grupos con un peso balanceado y luego permitir calibracion segun desempeno historico, disponibilidad de datos y region.

## Autor

Osmany Lorenzo Amaro
