from typing import Literal

from numpy import ndarray
from sklearn.svm import SVC

from dl_ml_library.abstract.model import Model


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

    def train(self, samples: ndarray, classes: ndarray) -> None:
        self.model.fit(X=samples, y=classes)

    def inference(self, samples: ndarray) -> ndarray:
        return self.model.predict(X=samples)
