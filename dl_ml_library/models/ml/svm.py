from typing import Literal

import numpy
from numpy import ndarray
from progress.bar import Bar
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from dl_ml_library.abstract.model import Model
from dl_ml_library.common.data import Data


class SVM(Model):
    def __init__(
        self,
        C: float = 1,
        kernel: Literal["rbf", "linear", "poly", "sigmoid"] = "rbf",
        degree: int = 3,
        gamma: Literal["scale", "auto"] | float = "scale",
        coef0: float = 0,
        cache_size: int = 200,
        max_iter: int = -1,
        decision_function_shape: Literal["ovo", "ovr"] = "ovr",
        break_ties: bool = False,
        random_state: int = 42,
    ) -> None:
        self.randomState: int = random_state
        self.model: SVC = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=True,
            probability=False,
            tol=0.001,
            cache_size=cache_size,
            class_weight=None,
            verbose=False,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=self.randomState,
        )

    def train(self, data: Data) -> None:
        self.model.fit(X=data.trainingDF_transformed_samples, y=data.trainingDF_classes)
        # Add metrics

    def inference(self, data: Data) -> None:
        predictions = self.model.predict(X=data.testingDF_transformed_samples)
        # Add metrics
        print(predictions)
