import os,sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GridSearchCV
import pandas as pd

class GridSearchHelper():
    def __init__(self):
        print("GridSearchHelper Created")

        self.gridSearchCV = None
        self.clf_and_params = list()

        self._initalize_clf_and_params()

    def _initalize_clf_and_params(self):

        clf = KNeighborsClassifier()
        params = {
            'n_neighbors': [5,7,9,11,13,15],
            'leaf_size': [1,2,3,5],
            'weights': ['uniform', 'distance']
        }
        self.clf_and_params.append((clf, params))

        clf = SVC()
        params = [
                  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
                 ]
        self.clf_and_params.append((clf, params))

        clf=LogisticRegression()
        params={'penalty':['l1', 'l2'],
                'C':np.logspace(0, 4, 10)
                }
        self.clf_and_params.append((clf, params))

        clf=DecisionTreeClassifier()
        params={'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
          'min_samples_leaf':[1],
          'random_state':[123]}
        #Because of depricating warning for Decision Tree which is not appended.
        #But it give high competion accuracy score. You can append when you run the kernel
        self.clf_and_params.append((clf,params))

        clf = RandomForestClassifier()
        params = {
            'n_estimators': [4, 6, 9],
            'max_features': ['log2', 'sqrt', 'auto'],
            'criterion': ['entropy', 'gini'],
            'max_depth': [2, 3, 5, 10],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 5, 8]
        }

        self.clf_and_params.append((clf, params))

    def fit_predict_save(self, X_train, X_test, y_train, submission_id, strategy_type):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.submission_id = submission_id
        self.strategy_type = strategy_type

        clf_and_params = self.get_clf_and_params()
        models = []

        self.results = {}
        for clf, params in clf_and_params:
            self.current_clf_name = clf.__class__.__name__
            grid_search_clf = GridSearchCV(clf, params, cv=5)
            grid_search_clf.fit(self.X_train, self.y_train)

            self.Y_pred = grid_search_clf.predict(self.X_test)
            clf_train_acc = round(grid_search_clf.score(self.X_train, self.y_train) * 100, 2)
            print(self.current_clf_name, "trained and used for prediction on test data...")
            self.results[self.current_clf_name] = clf_train_acc

            # for ensemble
            models.append(clf)

            self.save_result()
            print()

        """
        voting_clf=VotingClassifier(models)
        voting_clf.fit(self.X_train, self.y_train)
        self.Y_pred=voting_clf.predict(self.X_test)
        self.current_clf_name = clf.__class__.__name__
        clf_train_acc = round(voting_clf.score(self.X_train, self.y_train) * 100, 2)
        print(self.current_clf_name, " train accuracy:", clf_train_acc)
        self.save_result()
        """

    def show_result(self):
        for clf_name, train_acc in self.results.items():
            print("{} train accuracy is {:.3f}".format(clf_name, train_acc))

    def save_result(self):
        submission = pd.DataFrame({
            "PassengerId": self.submission_id,
            "Survived": self.Y_pred
        })

        file_name="{}_{}.csv".format(self.strategy_type,self.current_clf_name.lower())
        submission.to_csv(os.path.join('/workspace/output',file_name), index=False)

        print("Submission saved as {}".format(file_name))

    def get_clf_and_params(self):
        return self.clf_and_params

    def add(self, clf, params):
        self.clf_and_params.append((clf, params))
