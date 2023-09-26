from dl_ml_library.common.data import Data
from dl_ml_library.datasets.uci.iris import Iris
from dl_ml_library.preprocessors.scalers import BasicScaler


def main() -> None:
    scaler: BasicScaler = BasicScaler()

    dataset: Iris = Iris()
    dataset.download()

    data: Data = dataset.load()
    data.trainTestValidationSplit()

    scaler.processData(data=data)

    print(type(data.validationDF_transformed_samples))
    print(data.validationDF_transformed_samples)


if __name__ == "__main__":
    main()
