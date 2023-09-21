from abc import ABC, abstractclassmethod


class Model(ABC):
    @abstractclassmethod
    def train(self) -> None:
        pass

    @abstractclassmethod
    def inference(self) -> None:
        pass
