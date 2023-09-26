from sklearn.svm import SVC

from dl_ml_library.abstract.model import Model


class SVM(Model):
    def __init__(self) -> None:
        self.model: SVC = SVC()

    def train(self) -> None:
        pass

    def inference(self) -> None:
        pass
