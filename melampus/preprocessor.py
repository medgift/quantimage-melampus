import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, OrdinalEncoder


class MelampusPreprocessor(object):
    """
    This is the preprocessor class for melampus platform. It process data in order to be standarized and normalized.
    It also provides the method for dimensionaliry reduction and the removal of high correlated features.
    Melampus Preprocessor accepts a csv file with the **column names must be included**.
    The name of the target variable can also be given as an optional parameter in case that the target variable is included in the csv file.

    :param filename: The name of the csv file that includes the data
    :type filename: str, required
    :param target_col: name of the target variable if included in the csv dataset, defaults to None
    :type target_col: str, opional

    The transformed datasets are stored and can be accessed on pre.data object in numpy array format
    """

    def __init__(self, filename: str = None, dataframe: pd.DataFrame = None, target_col=None, outcomes=[],
                 id_names_map = {'patient_id' : 'PatientID', 'modality' : 'Modality'}):

        self.id_names_map = id_names_map
        self.id_names = list(id_names_map.keys())
        self.id_column_names = list(id_names_map.values())

        # check for filename vs dataframe and process
        self.filename = filename
        if self.filename and not isinstance(dataframe, pd.DataFrame):
            dataframe  = self._read_csv(self.filename)
        elif isinstance(dataframe, pd.DataFrame) and not self.filename:
            pass
        else:
            raise Exception("Specifify one of 'filename' / 'dataframe'")

        self.ids, data_tmp = self._split_columns_from_data(dataframe, self.id_column_names)

        # check for target_col vs outcomes and process
        self.target_col = target_col
        if self.target_col is not None and len(outcomes)==0:
            self.outcomes, data_tmp = self._split_columns_from_data(data_tmp, [self.target_col])
        elif self.target_col is None and len(outcomes)>0:
            if len(outcomes) != len(self.data):
                raise Exception("Iosistent number of cases between outcomes (%i) and samples (%i)"%(len(self.outcomes,
                                                                                                    self.num_cases)))
            else:
                self.outcomes = outcomes
        else:
            raise Exception("Specify one of 'target_col' / 'outcomes'")

        # assign cleaned data table
        self.data = data_tmp

        # compute stats
        self.num_cases = len(self.ids)
        self.num_cases_in_each_class = {}
        for el in self.outcomes:
            self.num_cases_in_each_class[el] = self.outcomes.count(el)

    def _read_csv(self, filename: str = None):
        if not filename:
            filename = self.filename
        try:
            self.data = pd.read_csv(filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(e)

    def _split_columns_from_data(self, dataframe: pd.DataFrame, column_names: list):
        print(column_names)
        data_sel = None
        data_remaining = dataframe
        try:
            data_sel = dataframe[column_names]
            data_remaining = dataframe.drop(columns=column_names)
        except KeyError as e:
            print(e)
        return data_sel, data_remaining






    def standarize_data(self):
        """
        Standarization of the data using scikit-learn ``StandardScaler``.
        """

        scaler = StandardScaler().fit(self.data)
        self.data = scaler.transform(self.data)

    def normalize_data(self):
        """
        Normalize data with L2 norm using ``normalize`` method from scikit-learn
        """

        self.data = normalize(self.data)

    def dimensionality_reduction(self, num_components: int):
        """
        Reduce the amount of features into a new feature space using Principal Component Analysis.

        :param num_components: number of dimentions in the new feature space
        :type num_components: int, required
        """

        pca = PCA(n_components=num_components)
        self.data = pca.fit_transform(self.data)
