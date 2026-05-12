"""Ejemplo minimo para descargar, combinar y consultar el multimodelo."""

from lluvia_multimodelo import MultiModeloMeteorologico


def main():
    modelo = MultiModeloMeteorologico()
    modelo.configurar(fecha="20260511", corrida="06", acumulado="f024-f144")
    modelo.cargar_todos_los_modelos()

    ds_multi = modelo.crear_multimodelo(
        pesos={
            "gfs": 0.25,
            "graphcast": 0.25,
            "ifs": 0.25,
            "aifs": 0.25,
        }
    )
    print(ds_multi)
    modelo.comparar_punto(lat=40.0, lon=-100.0)


if __name__ == "__main__":
    main()
