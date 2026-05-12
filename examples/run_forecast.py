"""Minimal example for downloading, blending, and querying the multi-model."""

from rainfall_multimodel import RainfallMultiModel


def main():
    model = RainfallMultiModel()
    model.configure(date="20260511", run="06", accumulation="f024-f144")
    model.load_all_models()

    multi_model = model.create_multimodel(
        weights={
            "gfs": 0.25,
            "ecmwf_ifs": 0.25,
            "ecmwf_aifs": 0.25,
            "gfs_graphcast": 0.25,
        }
    )
    print(multi_model)
    model.compare_point(latitude=40.0, longitude=-100.0)


if __name__ == "__main__":
    main()
