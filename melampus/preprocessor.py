import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, OrdinalEncoder


class MelampusPreprocessor(object):
    '''
    This is the preprocessor class for melampus platform. It process data in order to be standarized and normalized.
    It also provides the method for dimensionaliry reduction and the removal of high correlated features.
    Melampus Preprocessor accepts a csv file with the **column names must be included**. Also, the dataset must contain a separate column named exactly **'PatientID'** with the samples ids.
    The name of the target variable can also be given as an optional parameter in case that the target variable is included in the csv file.
    :param filename: The name of the csv file that includes the data
    :type filename: str, required
    :param target_col: name of the target variable if included in the csv dataset, defaults to None
    :type target_col: str, opional

    The transformed datasets are stored and can be accessed on pre.data object in numpy array format
    '''
    def __init__(self, filename: str, target_col=None):
        self.filename = '../' + filename
        self.target_col = target_col
        self.outcomes = []
        self.ids = pd.Series
        self.data = pd.DataFrame
        self.process_data_from_csv()

        if target_col is not None:
            self.identify_outcomes()

    def process_data_from_csv(self):
        '''
        Herein, the data are extracted from the csvfile and are transformed into a pandas DataFrame. The dataframe is stored
        into the self.data object.

        :raise Exception: If the path of csvfile is not valid or not found.
        :raise KeyError: If 'PatientID' column is not provided
        '''
        try:
            df = pd.read_csv(self.filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(e)

        # store patient ID in another list
        try:
            self.ids = df['PatientID']
            df = df.drop('PatientID', axis=1)  # delete column with Ids
            self.data = df.dropna(axis=1)  # delete columns with nans
        except KeyError as e:
            raise KeyError('Patient ID column not found {}'.format(e))

    def identify_outcomes(self):
        '''
        Herein, the outcomes are extracted from a column in the csvfile. This method is called only if ``target_col``
        parameter is provided which is the column with the outcomes in the csv file.
        '''
        self.outcomes = self.data[self.target_col].to_frame(self.target_col)
        self.data = self.data.drop(self.target_col, axis=1)

    def standarize_data(self):
        '''
        Standarization of the data using scikit-learn ``StandardScaler``.
        '''
        scaler = StandardScaler().fit(self.data)
        self.data = scaler.transform(self.data)

    def normalize_data(self):
        '''
        Normalize data with L2 norm using ``normalize`` method from scikit-learn
        '''
        self.data = normalize(self.data)

    def dimensionality_reduction(self, num_components: int):
        '''
        Reduce the amount of features into a new feature space using Principal Component Analysis.

        :param num_components: number of dimentions in the new feature space
        :type num_components: int, required
        '''
        pca = PCA(n_components=num_components)
        self.data = pca.fit_transform(self.data)