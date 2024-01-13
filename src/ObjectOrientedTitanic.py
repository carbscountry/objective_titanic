import os
import pandas as pd

from PreprocessStrategy import preprocessStrategy
from Visualizer import Visualizer
from GridSearchHelper import GridSearchHelper
from Information import Information




class ObjectOrientedTitanic:
    def __init__(self,train,test):
        """

        :param train: train data will be used for modelling
        :param test:  test data will be used for model evaluation

        """
        print("ObjectOrientedTitanic object Created")

        # properties
        self.testPassengerID = test['PassengerId']
        self.number_of_train = train.shape[0]

        self.y_train = train['Survived']
        self.train = train.drop(['Survived'], axis=1)
        self.test = test

        #concat train and test data
        self.all_data = self._get_all_data()

        # Create instance of objects
        self._info = Information()
        self.preprocessStrategy = preprocessStrategy()
        self.visualizer=Visualizer()
        self.gridSearchHelper = GridSearchHelper()

    def _get_all_data(self):
        return pd.concat([self.train, self.test])

    def information(self):
        """
        using _info object gives summary about dataset
        :return:
        """
        self._info.info(self.all_data)

    def preprocessing(self, strategy_type):
        """
        Process data depend upon strategy type
        :param strategy_type: Preprocessing strategy type
        :return:
        """
        self.strategy_type=strategy_type

        self.all_data = self.preprocessStrategy.strategy(self._get_all_data(), strategy_type)

    def visualize(self, visualizer_type, number_of_features=None):

        self._get_train_and_test()

        if visualizer_type == "RadViz":
            self.visualizer.RandianViz(self.X_train,
                                       self.y_train,
                                       number_of_features)

    def machine_learning(self):
        self._get_train_and_test()

        self.gridSearchHelper.fit_predict_save(self.X_train,
                                               self.X_test,
                                               self.y_train,
                                               self.testPassengerID,
                                               self.strategy_type)

    def show_result(self):
        self.gridSearchHelper.show_result()

    def _get_train_and_test(self):
        """
        Split data into train and test datasets
        *return:
        """

        self.X_train = self.all_data[:self.number_of_train]
        self.X_test = self.all_data[self.number_of_train:]
