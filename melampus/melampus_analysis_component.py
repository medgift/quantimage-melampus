import pandas as pd
from melampus.database import FeatureDB, OutcomeDB
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFECV, SelectKBest, GenericUnivariateSelect, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

class MelampusAnalysisComponent(object):

    def __init__(self, name):
        self.name = name

    def __init__(self, name, feature_array=None, outcome_array=None, missing_values=''):

        self.feature_array = feature_array
        self.outcome_array = outcome_array
        self.name = name

    def _has_data(self, name):
        status = False
        if (hasattr(self, name)):
            if name == 'feature_array':
                array = self.feature_array
            elif name == 'outcome_array':
                array = self.outcome_array
            if (array is None):
                status = False
            else:
                status = True
        else:
            status = False
        return status

    def has_feature_data(self):
        return self._has_data(name='feature_array')

    def has_outcome_data(self):
        return self._has_data(name='outcome_array')

    def _set_data(self, name, data):
        if name == 'feature_array':
            array = self.feature_array
        elif name == 'outcome_array':
            array = self.outcome_array
        if not self._has_data(name):
            array = data
        else:
            print("Already has '%s', skip" % name)

    def set_feature_array(self, data):
        self._set_data('feature_array', data)

    def set_outcome_array(self, data):
        self._set_data('outcome_array', data)

    def _get_data(self, name):
        if self._has_data(name):
            if name == 'feature_array':
                array = self.feature_array
            elif name == 'outcome_array':
                array = self.outcome_array
            return array
        else:
            raise Exception("Data '%s' not available")

    def get_feature_array(self):
        return self._get_data('feature_array')

    def get_outcome_array(self):
        return self._get_data('outcome_array')

    def get_num_features(self):
        return self.get_feature_array().shape[1]

    def get_num_samples(self):
        return self.get_feature_array().shape[0]

    def _get_indices_from_score(self, scores, ascending=True):
        if len(scores) != self.get_num_features():
            raise Exception("Expect same number of ranking scores as number of features")
        indices_sorted_ascending = np.argsort(scores, axis=0)  # default sorting order
        if ascending:
            indices_sorted = indices_sorted_ascending
        else:
            indices_sorted = indices_sorted_ascending[::-1]
        return indices_sorted


    def _return_as_array(self, indices):
        return self.get_feature_array()[:, indices]

    def _return(self, scores=None, indices=None, ascending=True, return_type='idx'):
        """
           Helper function to sort data array/frame/columns.
           :param scores: Scoring values by which features are to be ordered. This should be an array or list of identical
           length as number of features.
           :param ascending: If true, sorting is 'ascending' otherwise 'descending'
           :param return_type: One of 'idx' (indices), 'arrary' (data array)
           :return: List of indices or feature names, data array, dataframe
       """
        if (scores is not None) and (indices is  None):
            indices = self._get_indices_from_score(scores=scores, ascending=ascending)
        elif (scores is None) and (indices is not None):
            pass
        else:
            raise Exception("Provide one of 'scores' or 'indices'")

        if return_type == 'idx':
            return indices
        elif return_type == 'array':
            return self.get_feature_array()[:, indices]
        else:
            raise Exception("Return type '%s' undefined. Choose one of 'idx', 'array'")

class FeatureRanker(MelampusAnalysisComponent):

    def __init__(self, name):
        super().__init__(name)

    def __init__(self, name, feature_array=None, outcome_array=None):
        super().__init__(name, feature_array, outcome_array)

    def rank_by_univariate_f(self, ascending=True, return_type='idx'):
        """
        Ranks features by F value from univariate ANOVA.
        -> one F value per feature
        """
        F, pval = f_classif(self.get_feature_array(), self.get_outcome_array())
        return self._return(scores=F, ascending=ascending, return_type=return_type)


class FeatureSelector(MelampusAnalysisComponent):

    def __init__(self, name):
        super().__init__(name)

    def __init__(self, name, feature_array=None, outcome_array=None):
        super().__init__(name, feature_array, outcome_array)

    def select_by_univariate_f(self, mode='k_best', param=5, return_type='idx'):
        """
        Returns dataset including only 'best' features selected by F score (univariate, for classification problem).

        Selection approaches:
        :param mode: selection mode; one of: ‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’; default is 'k_best'
        :param params: float or int depending on feature selection mode; default is 5 for '5 k-best'
         For more details see scikit-learn feature_selection.GenericUnivariateSelect
        :return: transformed dataset, only containing selected features
        """
        sel = GenericUnivariateSelect(f_classif, mode=mode, param=param)
        sel.fit_transform(self.get_feature_array(), self.get_outcome_array())
        selected_features_idxs = sel.get_support(indices=True)
        return self._return(indices=selected_features_idxs, return_type=return_type)

    def variance_threshold(self, p_val=None, return_type='idx'):
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
            feature_df = pd.DataFrame(self.get_feature_array())
            sel.fit_transform(feature_df)
            selected_features_idxs = sel.get_support(indices=True)
            return self._return(indices=selected_features_idxs, return_type=return_type)
        except Exception as e:
            raise Exception(
                "feature_selector-variance_threshold: EXCEPTION: {}".format(e)
            )

    def drop_correlated_features(self, score: float, metric: str, return_type='idx'):
        """
        Herein this method deletes all the high correlated features based on the provided correlation score and a specific metric.

        :param score: correlation score
        :type score: float, required
        :param metric: {‘pearson’, ‘kendall’, ‘spearman’} or callable function
        :type metric: str, required
        :return: pandas dataframe
        """
        np.corrcoef()
        df = pd.DataFrame(self.get_feature_array())
        corr_matrix = df.corr(method=metric).abs()  # correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        # Find index of feature columns with correlation greater than a defined score and then drop these features
        cols_to_keep = [column for column in upper.columns if any(upper[column] <= score)]
        selected_features_idxs = [cols_to_keep.index(col) for col in cols_to_keep]
        return self._return(indices=selected_features_idxs, return_type=return_type)

    def identify_correlated_features_with_target_variable(self, score: float, metric: str):
        """
        With this method we identify all features that are high correlated with the outcome variable.
        It should be used only for regression tasks. The target variable **must be included into the dataset**.

        :param score: correlation score
        :type score: float, required
        :param metric: {‘pearson’, ‘kendall’, ‘spearman’} or callable function
        :type metric: str, required
        :return: The names of the relevant correlated features
        """
        outcome_name = 'outcome'
        df = pd.DataFrame(self.get_feature_array())
        df[outcome_name] = self.get_outcome_array()
        corr_matrix = df.corr(method=metric).abs()  # correlation matrix
        # Correlation with output variable
        cor_target = abs(corr_matrix[outcome_name])
        corr_feats = cor_target[cor_target > score]
        relevant_feats = corr_feats.drop(outcome_name)  # remove target variable
        selected_features_idxs = [cols_to_keep.index(col) for col in cols_to_keep]
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

