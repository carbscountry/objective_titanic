class Information():

    def __init__(self):
        """
        This class give some brief information about the datasets.
        Information introduced in R language style
        """
        print("Information object created")

    def _get_missing_values(self,data):
        """
        Find missing values of given datad
        :param data: checked its missing value
        :return: Pandas Series object
        """
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)

        #Returning missing values
        return missing_values

    def info(self,data):
        """
        print feature name, data type, number of missing values and ten samples of
        each feature
        :param data: dataset information will be gathered from
        :return: no return value
        """
        feature_dtypes=data.dtypes
        self.missing_values=self._get_missing_values(data)

        print("=" * 50)

        print("{:16} {:16} {:25} {:16}".format("Feature Name".upper(),
                                            "Data Format".upper(),
                                            "# of Missing Values".upper(),
                                            "Samples".upper()))
        for feature_name, dtype, missing_value in zip(self.missing_values.index.values,
                                                      feature_dtypes[self.missing_values.index.values],
                                                      self.missing_values.values):
            print("{:18} {:19} {:19} ".format(feature_name, str(dtype), str(missing_value)), end="")
            for v in data[feature_name].values[:10]:
                print(v, end=",")
            print()

        print("="*50)
