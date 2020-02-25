from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris


class FeatureSelector(object):
    def variance_threshold(self):

        p = 0.8  # hardcoded for the moment
        thres = p * (1 - p)
        pass

    def pearson_corr(self):
        pass

    def rfe(self):
        pass
