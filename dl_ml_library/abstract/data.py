from abc import ABC, abstractmethod

from pandas import DataFrame
from sklearn.model_selection import train_test_split


class Data(ABC):
    def __init__(self, df: DataFrame) -> None:
        self.df: DataFrame = df
        self.trainingDF: DataFrame = None
        self.testingDF: DataFrame = None
        self.validationDF: DataFrame = None

    def trainTestValidationSplit(
        self,
        trainSplit: float = 0.7,
        validationSplit: float = 0.15,
        testSplit: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.trainingDF, self.testingDF = train_test_split(
            self.df,
            test_size=testSplit,
            random_state=seed,
            shuffle=True,
        )
        self.trainingDF, self.validationDF = train_test_split(
            self.trainingDF,
            test_size=validationSplit,
            random_state=seed,
            shuffle=True,
        )
