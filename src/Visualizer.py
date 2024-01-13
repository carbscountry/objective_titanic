import matplotlib.pyplot as plt

from yellowbrick.features import RadViz

class Visualizer:

    def __init__(self) -> None:
        print("Visualizer Created")

    def RandianViz(self, X, y, number_of_features):
        if number_of_features is None:
            features = X.columns.values

        else:
            features = X.columns.values[:number_of_features]

        fig, ax = plt.subplots(figsize=(15,12))
        radViz=RadViz(classes=['survived', 'not survived'], features=features)

        radViz.fit(X, y)
        radViz.transform(X)
        radViz.poof()
