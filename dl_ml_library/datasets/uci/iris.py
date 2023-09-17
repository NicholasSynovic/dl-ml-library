from io import BytesIO
from zipfile import ZipFile

from requests import Response, get

from dl_ml_library.abstract.dataset import Dataset


class Iris(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.homeURL: str = "https://archive.ics.uci.edu/dataset/53/iris"
        self.downloadURL: str = "https://archive.ics.uci.edu/static/public/53/iris.zip"

    def download(self) -> None:
        resp: Response = get(url=self.downloadURL, headers=self.requestHeaders)

        with open("iris.zip", "wb") as archive:
            archive.write(resp.content)
            archive.close()

        resp.close()

    def load():
        pass


i = Iris()
i.download()
