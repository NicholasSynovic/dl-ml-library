from sklearn.preprocessing import StandardScaler

from dl_ml_library.abstract.preprocessor import Preprocessor
from dl_ml_library.common.data import Data


class BasicScaler(Preprocessor):
    def __init__(
        self,
        withMean: bool = True,
        withStandardDeviation: bool = True,
    ) -> None:
        self.processor: StandardScaler = StandardScaler(
            with_mean=withMean,
            with_std=withStandardDeviation,
        )

    def processData(self, data: Data) -> None:
        self.processor.fit(X=data.trainingDF)
        data.trainingDF = self.processor.transform(X=data.trainingDF)
        data.testingDF = self.processor.transform(X=data.testingDF)
        data.validationDF = self.processor.transform(X=data.validationDF)
