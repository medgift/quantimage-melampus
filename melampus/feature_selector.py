from sklearn.feature_selection import VarianceThreshold, RFECV, SelectKBest, GenericUnivariateSelect, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from melampus.preprocessor import MelampusPreprocessor


class MelampusFeatureSelector(MelampusPreprocessor):
    def __init__(self, filename: str = None, dataframe: pd.DataFrame = None):
        if not filename:
            self.data = dataframe
        else:
            self.data = pd.read_csv(filename)

    """
    Melampus Feature Selector consists of three methods for identifying features based on the filter we want to apply.
    It inherits the inputs of :class:`melampus.preprocessor.MelampusPreprocessor`
    """

    def univariate_f_classification(self, mode='k_best', param=5):
        """
        Returns dataset including only 'best' features selected by F score (univariate, for classification problem).

        Selection approaches:
        :param mode: selection mode; one of: ‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’; default is 'k_best'
        :param params: float or int depending on feature selection mode; default is 5 for '5 k-best'
         For more details see scikit-learn feature_selection.GenericUnivariateSelect
        :return: transformed dataset, only containing selected features
        """
        sel = GenericUnivariateSelect(f_classif, mode=mode, param=param)
        sel.fit_transform(self.data, self.outcome)
        return self.data[self.data.columns[sel.get_support(indices=True)]]


    def variance_threshold(self, p_val=None):
        """
        It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features

        :param p_val: p_value for defining the threshold. default value: 0.8
        :return: transormed array of removed correlated features
        """

        p = 0.8
        if p_val:
            p = p_val

        thres = p * (1 - p)
        sel = VarianceThreshold(threshold=thres)
        try:
            # Perform the feature selection
            sel.fit_transform(self.data)

            # Return a DataFrame with remaining column names
            return self.data[self.data.columns[sel.get_support(indices=True)]], p
        except Exception as e:
            raise Exception(
                "feature_selector-variance_threshold: EXCEPTION: {}".format(e)
            )

    def drop_correlated_features(self, score: float, metric: str):
        """
        Herein this method deletes all the high correlated features based on the provided correlation score and a specific metric.

        :param score: correlation score
        :type score: float, required
        :param metric: {‘pearson’, ‘kendall’, ‘spearman’} or callable function
        :type metric: str, required
        :return: pandas dataframe
        """

        df = self.data
        corr_matrix = df.corr(method=metric).abs()  # correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        # Find index of feature columns with correlation greater than a defined score and then drop these features
        to_drop = [column for column in upper.columns if any(upper[column] > score)]
        df = df.drop(df[to_drop], axis=1)
        return df

    def identify_correlated_features_with_target_variable(
        self, score: float, metric: str, target_var: str
    ):
        """
        With this method we identify all features that are high correlated with the outcome variable.
        It should be used only for regression tasks. The target variable **must be included into the dataset**.

        :param score: correlation score
        :type score: float, required
        :param metric: {‘pearson’, ‘kendall’, ‘spearman’} or callable function
        :type metric: str, required
        :param target_var: name of the target variable included in the csv dataset
        :type target_var: str, required
        :return: The names of the relevant correlated features
        """

        df = self.data.join(
            self.outcomes
        )  # Merge data with target variable into one dataframe
        corr_matrix = df.corr(method=metric).abs()  # correlation matrix
        # Correlation with output variable
        cor_target = abs(corr_matrix[target_var])
        corr_feats = cor_target[cor_target > score]
        relevant_feats = corr_feats.drop(target_var)  # remove target variable
        return relevant_feats

    def rfe(self):
        """
        Recursive feature elimination for selecting the features that contribute most to the target variable (most significant features).
        Specifically: A Logistic Regression estimator is trained on the initial set of features and the importance of each feature is
        obtained. the least important features are pruned from current set of features.
        That procedure is recursively repeated on the pruned set until the desired number of features to select is
        eventually reached.

        **This is a slow method. It should be used for small number of features (less than 20)**

        :return: the dataset with the final selected features
        """

        logreg = LogisticRegression()
        rfecv = RFECV(
            estimator=logreg, cv=StratifiedKFold(), scoring="accuracy", n_jobs=-1
        )
        try:
            rfecv.fit(self.data, self.outcomes)
            k = rfecv.n_features_
            return SelectKBest(k=k).fit_transform(self.data)
        except Exception as e:
            print("feature_selector-rfe: EXCEPTION: {}".format(e))
            pass
