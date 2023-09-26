from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Data:
    def __init__(self, df: DataFrame) -> None:
        self.df: DataFrame = df
        self.encoder: LabelEncoder = LabelEncoder()
        self.encoder.fit(y=self.df["Class"])

        self.trainingDF: DataFrame
        self.testingDF: DataFrame
        self.validationDF: DataFrame

        self.trainingDF_samples: DataFrame
        self.testingDF_samples: DataFrame
        self.validationDF_samples: DataFrame

        self.trainingDF_transformed_samples: ndarray
        self.testingDF_transformed_samples: ndarray
        self.validationDF_transformed_samples: ndarray

        self.trainingDF_classes: DataFrame
        self.testingDF_classes: DataFrame
        self.validationDF_classes: DataFrame

        self.trainingDF_encoded_classes: ndarray
        self.testingDF_encoded_classes: ndarray
        self.validationDF_encoded_classes: ndarray

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

        self.trainingDF_encoded_classes = self.encoder.transform(
            y=self.trainingDF_classes
        )
        self.testingDF_encoded_classes = self.encoder.transform(
            y=self.testingDF_classes
        )
        self.validationDF_encoded_classes = self.encoder.transform(
            y=self.validationDF_classes
        )
