import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, OrdinalEncoder


class Preprocessor(object):
    def __init__(self, filename: str, target_col= None):
        '''
        A preprocessor for image datasets.
        * dimensionality reduction
        * Standarization scaling
        * Normalization
        * Removal of high correlated features
        ''' ''
        self.filename = filename
        self.target_col = target_col
        self.outcomes = []
        self.ids = pd.Series
        self.data = pd.DataFrame
        self.process_data_from_csv()

        if target_col is not None:
            self.identify_outcomes()

    def process_data_from_csv(self):
        try:
            df = pd.read_csv(self.filename)
            # store patient ID in another list
            self.ids = df['PatientID']
            df = df.drop('PatientID', axis=1) # delete column with Ids
            self.data = df.dropna(axis=1) # delete columns with nans
        except Exception as e:
            print('error in processing csv file {}'.format(e))
            return ''

    def identify_outcomes(self):
        self.outcomes = self.data[self.target_col].to_frame(self.target_col)
        self.data = self.data.drop(self.target_col, axis=1)

    def standarize_data(self):
        scaler = StandardScaler().fit(self.data)
        self.data = scaler.transform(self.data)

    def normalize_data(self):
        '''
        normalize data with L2 norm
        '''
        self.data = normalize(self.data)

    def dimensionality_reduction(self, num_components=int):
        pca = PCA(n_components=num_components)
        self.data = pca.fit_transform(self.data)

    def encode_categorical_features(self):
        '''
        #TODO: on progress
        transform categorical features to integers [NOT FINISHED].
        e.g. (directly from sklearn): A person could have features ["male", "female"], ["from Europe", "from US", "from Asia"],
        ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]. Such features can be efficiently
        coded as integers, for instance ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3]
        while ["female", "from Asia", "uses Chrome"] would be [1, 2, 1].
        '''
        self.identify_categorical_features()
        return OrdinalEncoder().fit_transform(X).to_array()

    def identify_categorical_features(self):
        '''on going'''
        pass


#pre = Preprocessor(filename='../synthetic_data/output_L0_GTVL.csv')
#data_tr = pre.standarize_data()
#x=1