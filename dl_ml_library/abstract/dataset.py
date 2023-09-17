from abc import ABC, abstractmethod
from pathlib import Path

from dl_ml_library.abstract.data import Data


class Dataset(ABC):
    def __init__(self) -> None:
        self.requestHeaders: dict[str, str] = {
            "User-Agent": "NicholasSynovic/dl-ml-library",
        }

    @abstractmethod
    def download(self) -> Path:
        pass

    @abstractmethod
    def load(self) -> Data:
        pass
