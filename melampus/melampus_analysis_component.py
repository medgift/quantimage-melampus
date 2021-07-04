import pandas as pd
from melampus.database import FeatureDB, OutcomeDB
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFECV, SelectKBest, GenericUnivariateSelect, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, OrdinalEncoder

import melampus.tools as tools

class MelampusAnalysisComponent(object):

    def __init__(self, name):
        self.name = name

    def __init__(self, name, feature_array=None, outcome_array=None,
                 feature_names=None, sample_ids=None):

        self.feature_array = feature_array
        self.outcome_array = outcome_array
        self.feature_names = feature_names
        self.sample_ids    = sample_ids
        self.name = name

        # consistency check: number names agrees with data dimenasions

    def has_feature_names(self):
        status = False
        if (hasattr(self, 'feature_names')):
            name_list = self.feature_names
            if (name_list is None):
                status = False
            else:
                status = True
        else:
            status = False
        return status

    def has_sample_ids(self):
        status = False
        if (hasattr(self, 'sample_ids')):
            name_list = self.sample_ids
            if (name_list is None):
                status = False
            else:
                status = True
        else:
            status = False
        return status

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

    def _set_data(self, name, data, force=False):
        if name == 'feature_array':
            array = self.feature_array
            print("selected feature array")
        elif name == 'outcome_array':
            array = self.outcome_array
            print("selected outcome array")
        else:
            print("incorrect array name %s"%name)
        if not self._has_data(name):
            array = data
            print("assigning new array")
        elif self._has_data(name) and force:
            array = data
            print("overwrting array")
        else:
            print("Already has '%s', skip" % name)

    def set_feature_array(self, data, force=False):
        if not self.has_feature_data():
            self.feature_array = data
        elif self.has_feature_data() and force:
            self.feature_array = data
        else:
            print("Already has 'feature_array', skip")

    def set_outcome_array(self, data, force=False):
        if not self.has_outcome_data():
            self.outcome_array = data
        elif self.has_outcome_data() and force:
            self.outcome_array = data
        else:
            print("Already has 'outcome_array', skip")

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

    def get_feature_dataframe(self, include_ids=True, include_outcomes=False, outcome_name='outcome', id_name='index',
                              data=None):
        if self.has_feature_names():
            if data is None:
                df = pd.DataFrame(columns=self.feature_names, data=self.get_feature_array())
            else:
                df = pd.DataFrame(columns=self.feature_names, data=data)
        if include_outcomes:
            df[outcome_name] = self.get_outcome_array()
        if include_ids:
            df[id_name] = self.sample_ids
            df.set_index(id_name, inplace=True)
        return df

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

    def _return(self, scores=None, indices=None, ascending=True, return_as='idx'):
        """
           Helper function to sort data array/frame/columns.
           :param scores: Scoring values by which features are to be ordered. This should be an array or list of identical
           length as number of features.
           :param ascending: If true, sorting is 'ascending' otherwise 'descending'
           :param return_as: One of 'idx' (indices), 'arrary' (data array), 'names' (list of feature names),
                             'dataframe' (pandas dataframe)
           :return: List of indices or feature names, data array, dataframe
       """
        if (scores is not None) and (indices is  None):
            indices = self._get_indices_from_score(scores=scores, ascending=ascending)
        elif (scores is None) and (indices is not None):
            pass
        else:
            raise Exception("Provide one of 'scores' or 'indices'")

        if return_as== 'idx':
            return indices
        elif return_as== 'names':
            names_array = np.array(self.feature_names)
            names_array_new = names_array[indices]
            return names_array_new.tolist()
        elif return_as== 'array':
            return self.get_feature_array()[:, indices]
        elif return_as== 'dataframe':
            names_array = np.array(self.feature_names)
            names_array_new = names_array[indices]
            return self.get_feature_dataframe()[names_array_new.tolist()]
        else:
            raise Exception("Return type '%s' undefined. Choose one of 'idx', 'array'")

class FeatureSelector(MelampusAnalysisComponent):
    """
    Includes ranking.
    All included methods return a sequence of feature indices which can be translated into a reordering or selection
    from the original dataset. The underlying feature data itself is not manipulated.
    """

    def __init__(self, name):
        super().__init__(name)

    def __init__(self, name, feature_array=None, outcome_array=None, feature_names=None, sample_ids=None):
        super().__init__(name, feature_array, outcome_array, feature_names, sample_ids)

    def rank_by_univariate_f(self, ascending=True, return_as='idx'):
        """
        Ranks features by F value from univariate ANOVA.

        :param return_as: One of 'idx' (indices), 'arrary' (data array), 'names' (list of feature names),
                             'dataframe' (pandas dataframe)
        :return: List of indices or feature names, data array, dataframe
        """
        F, pval = f_classif(self.get_feature_array(), self.get_outcome_array())
        return self._return(scores=F, ascending=ascending, return_as=return_as)

    def select_by_univariate_f(self, mode='k_best', param=5, return_as='idx'):
        """
        Returns dataset including only 'best' features selected by F score (univariate, for classification problem).

        Selection approaches:
        :param mode: selection mode; one of: ‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’; default is 'k_best'
        :param params: float or int depending on feature selection mode; default is 5 for '5 k-best'
         For more details see scikit-learn feature_selection.GenericUnivariateSelect
        :param return_as: One of 'idx' (indices), 'arrary' (data array), 'names' (list of feature names),
                             'dataframe' (pandas dataframe)
        :return: List of indices or feature names, data array, dataframe
        """
        sel = GenericUnivariateSelect(f_classif, mode=mode, param=param)
        try:
            sel.fit_transform(self.get_feature_array(), self.get_outcome_array())
            selected_features_idxs = sel.get_support(indices=True)
            return self._return(indices=selected_features_idxs, return_as=return_as)
        except Exception as e:
            raise Exception(
                "FeatureSelector-select_by_univariate_f: EXCEPTION: {}".format(e)
                 )

    def drop_correlated_features_fast(self, thr=0.8, metric='pearson', return_as='idx'):
        """
        Removes a feature if its correlation coefficient with any of the other features exceeds the threshold value 'thr'.

        :param thr: upper limit on correlation score for a pair of features to be retained
        :param metric: {‘pearson’, ‘kendall’, ‘spearman’} or callable function
        :param return_as: One of 'idx' (indices), 'arrary' (data array), 'names' (list of feature names),
                             'dataframe' (pandas dataframe)
        :return: List of indices or feature names, data array, dataframe to be retained
        """
        data = self.get_feature_dataframe(include_outcomes=False)
        corr_matrix = data.corr(method=metric).abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        cols_to_drop = [column for column in upper.columns if any(upper[column] > thr)]
        cols_to_keep = list(set(data.columns).difference(cols_to_drop))
        columns = self.feature_names
        selected_features_idxs = [columns.index(col) for col in cols_to_keep]
        return self._return(indices=selected_features_idxs, return_as=return_as)


    def drop_correlated_features(self, thr=0.8, method='pearson', return_as='idx'):
        """
        Removes correlated features.
        This approach tries to avoid dropping features that are uncorrelated with the remaining features.
        Implementation is slow...

        :param thr: upper limit on correlation score for a pair of features to be retained
        :param method: {‘pearson’, ‘kendall’, ‘spearman’} or callable function
        :param return_as: One of 'idx' (indices), 'arrary' (data array), 'names' (list of feature names),
                             'dataframe' (pandas dataframe)
        :return: List of indices or feature names, data array, dataframe to be retained
        """
        data = self.get_feature_dataframe(include_outcomes=False)
        cols_to_drop = tools.find_correlated_features(data, method=method, cut=thr)
        cols_to_keep = list(set(data.columns).difference(cols_to_drop))
        columns = self.feature_names
        selected_features_idxs = [columns.index(col) for col in cols_to_keep]
        return self._return(indices=selected_features_idxs, return_as=return_as)


    def select_by_correlation_with_outcome(self, thre=0.8, method='pearson', return_as='idx'):
        """
        Returns features that are highly correlated (exceeding correlation threshold 'thr') with outcome variable.

        :param thr: lower limit of correlation coefficient with outcome for feature to be included
        :param method: {‘pearson’, ‘kendall’, ‘spearman’} or callable function
        :param return_as: One of 'idx' (indices), 'arrary' (data array), 'names' (list of feature names),
                             'dataframe' (pandas dataframe)
        :return: List of indices or feature names, data array, dataframe to be retained
        """
        outcome_name = 'outcome'
        df = self.get_feature_dataframe(include_outcomes=True, outcome_name=outcome_name)
        corr_matrix = df.corr(method=method).abs()
        corr_target     = corr_matrix[outcome_name]
        corr_features   = corr_target[corr_target > thre]
        cols_to_keep = corr_features.drop(outcome_name).index.tolist()
        columns = self.feature_names
        selected_features_idxs = [columns.index(col) for col in cols_to_keep]
        return self._return(indices=selected_features_idxs, return_as=return_as)

    # def rfe(self):
    #     """
    #     Recursive feature elimination for selecting the features that contribute most to the target variable (most significant features).
    #     Specifically: A Logistic Regression estimator is trained on the initial set of features and the importance of each feature is
    #     obtained. the least important features are pruned from current set of features.
    #     That procedure is recursively repeated on the pruned set until the desired number of features to select is
    #     eventually reached.
    #
    #     **This is a slow method. It should be used for small number of features (less than 20)**
    #
    #     :return: the dataset with the final selected features
    #     """
    #
    #     logreg = LogisticRegression()
    #     rfecv = RFECV(
    #         estimator=logreg, cv=StratifiedKFold(), scoring="accuracy", n_jobs=-1
    #     )
    #     try:
    #         rfecv.fit(self.get_feature_array(), self.get_outcome_array())
    #         k = rfecv.n_features_
    #         return SelectKBest(k=k).fit_transform(self.data)
    #     except Exception as e:
    #         print("feature_selector-rfe: EXCEPTION: {}".format(e))
    #         pass
    #

class FeatureManipulator(MelampusAnalysisComponent):
    """
    In contrast to the FeatureSelector, methods in this class change the feature values of the original dataset.
    """

    def __init__(self, name):
        super().__init__(name)

    def __init__(self, name, feature_array=None, outcome_array=None, feature_names=None, sample_ids=None):
        super().__init__(name, feature_array, outcome_array, feature_names, sample_ids)


    def standardize_features(self, return_as='array'):
        """
        Standarization of the data using scikit-learn ``StandardScaler`` -> zero mean, unit variance

        :param return_as: One of 'arrary' (data array), 'dataframe' (pandas dataframe)
        :param inplace: overwrites the internal data
        :return: data array, dataframe
        """
        scaler = StandardScaler(with_mean=True, with_std=True).fit(self.get_feature_array())
        data   = scaler.transform(self.get_feature_array())
        self.set_feature_array(data, force=True)
        if return_as=='array':
            return self.get_feature_array()
        elif return_as=='dataframe':
            return self.get_feature_dataframe()


    def normalize_features(self, return_as='array'):
        """
        Normalize data with L2 norm using ``normalize`` method from scikit-learn -> values between 0 and 1
        """
        data = normalize(self.get_feature_array(), axis=0) # normalize each feature independently
        self.set_feature_array(data, force=True)
        if return_as == 'array':
            return self.get_feature_array()
        elif return_as == 'dataframe':
            return self.get_feature_dataframe()


    # def dimensionality_reduction(self, num_components: int):
    #     """
    #     Reduce the amount of features into a new feature space using Principal Component Analysis.
    #
    #     :param num_components: number of dimentions in the new feature space
    #     :type num_components: int, required
    #     """
    #     pca = PCA(n_components=num_components)
    #     self.data = pca.fit_transform(self.data)

