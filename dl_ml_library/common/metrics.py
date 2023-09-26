from numpy import ndarray
from sklearn.metrics import *

from dl_ml_library.abstract.model import Model
from dl_ml_library.common.data import Data


class Metrics:
    def __init__(
        self,
        data: Data,
        useTestingData: bool = True,
        normalize: bool = True,
    ):
        self.useTestingData: bool = useTestingData
        self.normalize: bool = normalize

        if self.useTestingData:
            self.samples: ndarray = data.testingDF_transformed_samples
            self.classes: ndarray = data.testingDF_encoded_classes
        else:
            self.samples: ndarray = data.validationDF_transformed_samples
            self.samples: ndarray = data.validationDF_encoded_classes

    def compute(self, model: Model) -> None:
        predictions: ndarray = model.inference(samples=self.samples)
        accuracy: float = accuracy_score(
            y_true=None,
            y_pred=None,
            normalize=self.normalize,
        )
        pass
