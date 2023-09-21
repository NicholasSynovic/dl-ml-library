from abc import ABC, abstractmethod

from dl_ml_library.common.data import Data


class Preprocessor(ABC):
    @abstractmethod
    def processData(self, data: Data) -> None:
        pass
