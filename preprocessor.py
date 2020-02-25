from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, OrdinalEncoder

import numpy as np
import pandas
import csv

class Preprocessor(object):
    def __init__(self, data=pandas.DataFrame):
        '''
        A preprocessor for image datasets.
        * dimensionality reduction
        * Standarization scaling
        * Normalization
        * Removal of high correlated features
        '''
        self.data = data
        self.header = self.extract_header()

    def extract_header(self):
        header =
        pass

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
