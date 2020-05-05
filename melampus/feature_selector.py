from sklearn.feature_selection import VarianceThreshold, RFECV, SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
from melampus.preprocessor import MelampusPreprocessor


class MelampusFeatureSelector(MelampusPreprocessor):

    def variance_threshold(self, p_val=None):
        '''
        It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features
        :param p_val: p_value for defining the threshold. default value: 0.8
        :return: transormed array of removed correlated features
        '''
        p = 0.8
        if p_val:
            p = p_val

        thres = p * (1 - p)
        sel = VarianceThreshold(threshold=thres)
        try:
            return sel.fit_transform(self.data)
        except Exception as e:
            raise Exception("feature_selector-variance_threshold: EXCEPTION: {}".format(e))

    def drop_correlated_features(self, score: float, metric: str):
        '''
        :param score: correlation score
        :param metric: {‘pearson’, ‘kendall’, ‘spearman’} or callable function
        :return: dataframe
        '''
        df = self.data
        corr_matrix = df.corr(method=metric).abs()  # correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than a defined score and then drop these features
        to_drop = [column for column in upper.columns if any(upper[column] > score)]
        df = df.drop(df[to_drop], axis=1)
        return df

    def identify_correlated_features_with_target_variable(self, score: float, metric: str, target_var: str):
        '''
        It should be used only for regression tasks. The target variable must be included into the dataset.
        '''
        df = self.data.join(self.outcomes)  # Merge data with target variable into one dataframe
        corr_matrix = df.corr(method=metric).abs()  # correlation matrix
        # Correlation with output variable
        cor_target = abs(corr_matrix[target_var])
        corr_feats = cor_target[cor_target > score]
        relevant_feats = corr_feats.drop(target_var)  # remove target variable
        return relevant_feats

    def rfe(self):
        '''
        Select the features that contribute most to the target variable.
        Slow method. It should be used for small number of features (less than 20)
        :return: the selected features
        '''
        logreg = LogisticRegression()
        rfecv = RFECV(estimator=logreg, cv=StratifiedKFold(), scoring='accuracy', n_jobs=-1)
        try:
            rfecv.fit(self.data, self.outcomes)
            k = rfecv.n_features_
            return SelectKBest(k=k).fit_transform(self.data)
        except Exception as e:
            print("feature_selector-rfe: EXCEPTION: {}".format(e))
            pass
