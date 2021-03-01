from sklearn.feature_selection import f_classif
import numpy as np

from melampus.melampus_base import Melampus


class MelampusFeatureRank(Melampus):

    def _return_by_score(self, scores, ascending=True, return_type='idx'):
        """
        Helper function to sort data array/frame/columns.
        :param scores: Scoring values by which features are to be ordered. This should be an array or list of identical
        length as number of features.
        :param ascending: If true, sorting is 'ascending' otherwise 'descending'
        :param return_type: One of 'idx' (indices), 'names' (feature/column names), 'arrary' (data array), 'dataframe' (dataframe)
        :return: List of indices or feature names, data array, dataframe
        """
        if len(scores)!=self.num_features:
            raise Exception("Expect same number of ranking scores as number of features")

        indices_sorted_ascending = np.argsort(scores, axis=0) # default sorting order
        if ascending:
            indices_sorted = indices_sorted_ascending
        else:
            indices_sorted = indices_sorted_ascending[::-1]

        if return_type=='idx':
            return indices_sorted
        elif return_type=='names':
            return self.data_columns[indices_sorted]
        elif return_type=='array':
            return self.get_data_as_array()[:, indices_sorted]
        elif return_type=='dataframe':
            return self.get_data_as_dataframe()[self.data_columns[indices_sorted]]
        else:
            raise Exception("Return type '%s' undefined. Choose one of 'idx', 'sol_names', 'array', 'dataframe'")


    def rank_by_univariate_f(self, **kwargs):
        """
        Ranks features by F value from univariate ANOVA.
        -> one F value per feature
        """
        F, pval = f_classif(self.data, self.outcomes)
        return self._return_by_score(F, **kwargs)