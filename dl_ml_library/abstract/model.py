from abc import ABC, abstractclassmethod

from numpy import ndarray


class Model(ABC):
    @abstractclassmethod
    def train(self, samples: ndarray) -> None:
        pass

    @abstractclassmethod
    def inference(self, samples: ndarray) -> None:
        pass
