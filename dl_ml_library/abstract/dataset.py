from abc import ABC, abstractmethod
from pathlib import Path

from . import Data


class Dataset(ABC):
    @abstractmethod
    def download(self) -> Path:
        pass

    @abstractmethod
    def load(self) -> Data:
        pass
