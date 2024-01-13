import pandas as pd
from sklearn.preprocessing import LabelEncoder

class preprocess():
    def __init__(self):
        print("Preprocessing data...")

    def fillna(self, data, fill_strategies):
        for column, strategy in fill_strategies.items():
            if strategy == 'None':
                data[column] = data[column].fillna('None')
            elif strategy == 'Zero':
                data[column] = data[column].fillna(0)
            elif strategy == 'Mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'Mean':
                data[column] = data[column].fillna(data[column].mean())
            elif strategy == 'Median':
                data[column] = data[column].fillna(data[column].median())
            else:
                print("{}: There is no such thing as preprocess strategy".format(strategy))

        return data

    def drop(self, data , drop_strategies):
        for column, strategy in drop_strategies.items():
            data = data.drop(column, axis=strategy)

        return data

    def feature_engineering(self, data, feature_engineering_strategies=1):
        if feature_engineering_strategies == 1:

            return self._feature_engineering_1(data)

    def _feature_engineering_1(self, data):
        data = self._base_feature_engineering(data)

        data.FareBin = pd.qcut(data.Fare,4,labels=[0,1,2,3])

        data.AgeBin = pd.cut(data.Age.astype(int),5,labels=[0,1,2,3,4])

        drop_strategy = {'Age': 1, # 1は列方向を示す
                         'Fare': 1,
                         'Name': 1
                         }

        data = self.drop(data, drop_strategy)

        return data

    def _base_feature_engineering(self, data):
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

        data['IsAlone'] = 1
        data['IsAlone'].loc[data['FamilySize'] > 1] = 0

        data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

        data['FareBin'] = pd.qcut(data['Fare'], 4)

        data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

        return data

    def _label_encoding(self,data):
        label_Encoder = LabelEncoder()
        for column in data.columns.values:
            if data[column].dtype == 'int64' or data[column].dtype == 'float64':
                continue
            label_Encoder.fit(data[column])
            data[column] = label_Encoder.transform(data[column])
        return data

    def _get_dummies(self, data, prefoerd_columns=None):

        if prefoerd_columns is None:
            columns = data.columns.values
            non_dummies = None
        else:
            non_dummies = [col for col in data.columns.values if col not in prefoerd_columns]

            columns = prefoerd_columns

        dummies_data = [pd.get_dummies(data[col], prefix=col) for col in columns]

        if non_dummies is not None:
            for non_dummy in non_dummies:
                dummies_data.append(data[non_dummy])

        return pd.concat(dummies_data, axis=1)
