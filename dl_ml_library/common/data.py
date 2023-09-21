from pandas import DataFrame
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, df: DataFrame) -> None:
        self.df: DataFrame = df
        self.trainingDF: DataFrame
        self.testingDF: DataFrame
        self.validationDF: DataFrame

        self.trainingDF_samples: DataFrame
        self.testingDF_samples: DataFrame
        self.validationDF_samples: DataFrame

        self.trainingDF_classes: DataFrame
        self.testingDF_classes: DataFrame
        self.validationDF_classes: DataFrame

    def trainTestValidationSplit(
        self,
        trainSplit: float = 0.7,
        validationSplit: float = 0.15,
        testSplit: float = 0.15,
        seed: int = 42,
    ) -> None:
        splitSum: float = trainSplit + validationSplit + testSplit
        if splitSum != 1:
            print("Waring: Splits do not equal 1.0")

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

        self.trainingDF_classes = self.trainingDF["Class"]
        self.testingDF_classes = self.testingDF["Class"]
        self.validationDF_classes = self.validationDF["Class"]

        self.trainingDF_samples = self.trainingDF.drop(columns="Class")
        self.testingDF_samples = self.testingDF.drop(columns="Class")
        self.validationDF_samples = self.validationDF.drop(columns="Class")
