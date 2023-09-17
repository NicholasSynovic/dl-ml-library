from requests import Response, get

from dl_ml_library.abstract.dataset import Dataset


class Iris(Dataset):
    def __init__(self) -> None:
        self.homeURL: str = "https://archive.ics.uci.edu/dataset/53/iris"
        self.downloadURL: str = "https://archive.ics.uci.edu/static/public/53/iris.zip"

    def download(self) -> None:
        print(self.requestHeaders)
        resp: Response = get(url=self.downloadURL, headers=self.requestHeaders)
        return None


i = Iris()
i.download()
