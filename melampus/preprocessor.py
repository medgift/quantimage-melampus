from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, OrdinalEncoder

import numpy as np
import pandas as pd
import csv


class Preprocessor(object):
    def __init__(self, filename: str, target_col= str):
        '''
        A preprocessor for image datasets.
        * dimensionality reduction
        * Standarization scaling
        * Normalization
        * Removal of high correlated features
        '''

        self.ids, self.data = self.process_data_from_csv(filename=filename)
        self.outcomes = self.data[target_col]


    def extract_outcomes(self,target_col: str):
        df =self.data
        return self.data
        pass

    def process_data_from_csv(self, filename: str):
        try:
            df = pd.read_csv(filename)
            # store patient ID in another list
            patient_IDs = df['PatientID']
            df = df.drop('PatientID', axis=1)
            #TODO remove cols with nans
            return patient_IDs, df
        except Exception as e:
            print('error in processing csv file {}'.format(e))
            return ''

    def standarize_data(self):
        scaler = StandardScaler().fit(self.data)
        self.data = scaler.transform(self.data)
        #return scaler.transform(self.data)

    def normalize_data(self, norm=str):
        return normalize(self.data, norm=norm)

    def dimensionality_reduction(self, num_components=int):
        pca = PCA(n_components=num_components)
        return pca.fit_transform(self.data)

    def encode_categorical_features(self):
        '''
        transform categorical features to integers.
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