from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, OrdinalEncoder

import numpy as np
import pandas as pd
import csv

class Preprocessor(object):
    def __init__(self, filename: str):
        '''
        A preprocessor for image datasets.
        * dimensionality reduction
        * Standarization scaling
        * Normalization
        * Removal of high correlated features
        '''

        self.header, self.data = self.import_data_from_csvfile(filename=filename)

    def import_data_from_csvfile(self, filename: str):
        try:
            df = pd.read_csv(filename)

        except Exception as e:
            print('error in reading file {}'.format(e))
            return ''

        return data

    def standarize_data(self):
        scaler = StandardScaler().fit(self.data)
        return scaler.transform(self.data)

    def normalize_data(self, norm=str):
        return normalize(self.data, norm=norm)

    def dimensionality_reduction(self, num_components=int):
        pca = PCA(n_components=num_components)
        return pca.fit_transform(self.data)

    @staticmethod
    def encode_categorical_features():
        '''
        transform categorical features to integers.
        e.g.:
        '''

        return OrdinalEncoder().fit_transform(X).to_array()

