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
        self.processor.fit(X=data.trainingDF_samples)
        data.trainingDF = self.processor.transform(X=data.trainingDF_samples)
        data.testingDF = self.processor.transform(X=data.testingDF_samples)
        data.validationDF = self.processor.transform(X=data.validationDF_samples)
