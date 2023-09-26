from dl_ml_library.common.data import Data
from dl_ml_library.datasets.uci.iris import Iris
from dl_ml_library.models.ml.svm import SVM
from dl_ml_library.preprocessors.scalers import BasicScaler


def main() -> None:
    scaler: BasicScaler = BasicScaler()

    dataset: Iris = Iris()
    # dataset.download()

    data: Data = dataset.load()
    data.trainTestValidationSplit()

    scaler.processData(data=data)

    svm: SVM = SVM()
    svm.train(data=data)
    svm.inference(data=data)


if __name__ == "__main__":
    main()
