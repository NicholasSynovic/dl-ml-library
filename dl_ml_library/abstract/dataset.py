from abc import ABC, abstractmethod
from os import makedirs
from os.path import expandvars
from pathlib import Path

from dl_ml_library.abstract.data import Data


class Dataset(ABC):
    def __init__(self) -> None:
        self.requestHeaders: dict[str, str] = {
            "User-Agent": "NicholasSynovic/dl-ml-library",
        }

        self.rootDownloadPath: Path = Path(
            expandvars(path="$HOME/.cache/dl-ml-datasets")
        )
        makedirs(name=self.rootDownloadPath, exist_ok=True)

    @abstractmethod
    def download(self) -> Path:
        pass

    @abstractmethod
    def load(self) -> Data:
        pass
