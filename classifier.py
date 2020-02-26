import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class PocClassifier(object):
    def __init__(self, data= np.ndarray, scaling= False, dim_red= (False, int)):
        self.classifier = LogisticRegression()
        if scaling:
            data = self.transform_training_set(data=data)

        if dim_red[0]:
            data = self.dimensionality_reduction(data=data, num_components=dim_red[1])

        self.train(data)
        self.assess_classifier()


    @staticmethod
    def transform_training_set(data):
        scaler = StandardScaler().fit(data)
        data_scaled = scaler.transform(data)
        return data_scaled

    @staticmethod
    def dimensionality_reduction(data= np.ndarray, num_components= int):
        pca = PCA(n_components=num_components)
        return pca.fit_transform(data)

    def train(self, data= np.ndarray):
        self.classifier.fit(data)

    def assess_classifier(self):
        scores = cross_val_score(self.classifier, X_scaled, y, cv=10)
        return scores.mean()
