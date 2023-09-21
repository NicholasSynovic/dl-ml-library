from os import listdir, makedirs
from pathlib import Path
from typing import List
from zipfile import ZipFile

from requests import Response, get

from dl_ml_library.abstract.dataset import Dataset


class Iris(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.homeURL: str = "https://archive.ics.uci.edu/dataset/53/iris"
        self.downloadURL: str = "https://archive.ics.uci.edu/static/public/53/iris.zip"
        self.downloadPath: Path = Path(self.rootDownloadPath, "iris")
        self.zipPath: Path = Path(self.downloadPath, "iris.zip")

        makedirs(name=self.downloadPath, exist_ok=True)

    def download(self) -> List[Path]:
        print(f"Downloading iris.zip to: {self.downloadPath}")
        resp: Response = get(url=self.downloadURL, headers=self.requestHeaders)

        with open(self.zipPath, "wb") as archive:
            archive.write(resp.content)
            archive.close()

        resp.close()

        print(f"Extracting data from: {self.zipPath}")
        with ZipFile(file=self.zipPath, mode="r") as zf:
            zf.extractall(path=self.downloadPath)
            zf.close()

        files: List[Path] = [
            Path(self.downloadPath, file) for file in listdir(path=self.downloadPath)
        ]
        return files

    def load():
        pass


i = Iris()
x = i.download()

print(x)
