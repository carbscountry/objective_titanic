from Processing import preprocess

class preprocessStrategy():
    """
    前処理のクラス
    """

    def __init__(self):
        self.data = None
        self._preprocessor = preprocess()

    def strategy(self, data, strategy_type="strategy1"):
        self.data = data
        if strategy_type == "strategy1":
            self._strategy1()
        elif strategy_type == "strategy2":
            self._strategy2()
        else:
            print("There is no such thing as preprocess strategy")

        return self.data

    def _bash_strategy(self):
        drop_strategy = {'PassengerId': 1,
                         'Ticket': 1,
                         'Cabin': 1
                         }
        self.data = self._preprocessor.drop(self.data, drop_strategy)

        fill_strategy = {
                            'Age': 'Median',
                            'Fare': 'Median',
                            'Embarked': 'Mode'
                        }

        self.data = self._preprocessor.fillna(self.data, fill_strategy)

        self.data = self._preprocessor.feature_engineering(self.data, 1)

        self.data = self._preprocessor._label_encoding(self.data)

    def _strategy1(self):
        self._bash_strategy()

        self.data = self._preprocessor._get_dummies(self.data,prefoerd_columns=[
            'Pclass', 'Sex', 'Parch', 'Embarked', 'Title', 'IsAlone'])

    def _strategy2(self):
        self._bash_strategy()

        self.data = self._preprocessor._get_dummies(self.data,prefoerd_columns=None)
