from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris


class FeatureSelector(object):
    def __init__(self, data):
        self.data = data

    def variance_threshold(self, p_val=int):
        '''
         It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features
        :param p_val: p_value for defining the threshold. default value: 0.8
        :return: transormed array of removed correlated features
        '''
        p = 0.8  # hardcoded for the moment
        if p_val:
            p = p_val

        thres = p * (1 - p)
        sel = VarianceThreshold(threshold=thres)
        return sel.fit_transform(self.data)

    def pearson_corr(self):

        pass

    def rfe(self):

        pass


fs = FeatureSelector()
x_tr = fs.variance_threshold(p_val=0.9)
x = 1
