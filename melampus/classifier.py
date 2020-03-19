import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from melampus.preprocessor import Preprocessor


class MelampusClassifier(object):
    def __init__(self, filename: str, target_col=str, scaling=False, dim_red=(False, 0)):
        self.filename = filename
        self.target_col = target_col
        self.scaling = scaling
        self.dim_red, self.num_components = [dim_red[0],dim_red[1]]
        self.data, self.outcomes = [], []
        self.classifier = object
        self.preprocess_data()

    def preprocess_data(self):
        pre = Preprocessor(filename=self.filename, target_col=self.target_col)
        if self.scaling:
            pre.standarize_data()

        if self.dim_red:
            pre.dimensionality_reduction(num_components=self.num_components)

        self.data = pre.data
        self.outcomes = pre.outcomes

    @staticmethod
    def transform_training_set(data):
        scaler = StandardScaler().fit(data)
        data_scaled = scaler.transform(data)
        return data_scaled

    @staticmethod
    def dimensionality_reduction(data=np.ndarray, num_components=int):
        pca = PCA(n_components=num_components)
        return pca.fit_transform(data)

    def train(self):
        self.classifier = LogisticRegression()
        self.classifier.fit(self.data, self.outcomes)
        return self.classifier

    def assess_classifier(self):
        scores = cross_val_score(self.classifier, X_scaled, y, cv=10)
        return scores.mean()

