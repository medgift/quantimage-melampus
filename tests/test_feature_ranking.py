import unittest
from melampus.feature_ranking import MelampusFeatureRank
import tests.config as config
import pandas as pd
import numpy as np


class TestMelampusFeatureRanl(unittest.TestCase):

    def setUp(self):
        self.path_to_data = config.path_to_test_data
        self.data_df = pd.read_csv(self.path_to_data)
        self.outcomes = config.gen_random_outcome(self.data_df, n_classes=2)
        self.mfr = MelampusFeatureRank(filename=self.path_to_data, outcomes=self.outcomes)

    def test_return_by_score(self):
        rnd_score = np.random.random(self.mfr.num_features)
        idx_min = 5
        idx_max = 10
        rnd_score[idx_min] = min(rnd_score) - 1
        rnd_score[idx_max] = max(rnd_score) + 1
        # expected to raise exception -> length inconsistent
        self.assertRaises(Exception, self.mfr._return_by_score, rnd_score[:-1])
        # check outputs -- IDX
        indices_ascending = self.mfr._return_by_score(rnd_score, ascending=True, return_type='idx')
        self.assertTrue(indices_ascending[0]==idx_min and indices_ascending[-1]==idx_max)
        indices_decending = self.mfr._return_by_score(rnd_score, ascending=False, return_type='idx')
        self.assertTrue(indices_decending[0]==idx_max and indices_decending[-1]==idx_min)
        # check outputs -- COLUMN NAMES
        name_min = self.mfr.data_columns[idx_min]
        name_max = self.mfr.data_columns[idx_max]
        names_ascending = self.mfr._return_by_score(rnd_score, ascending=True, return_type='names')
        self.assertTrue(names_ascending[0] == name_min and names_ascending[-1] == name_max)
        # check outputs -- ARRAY
        values_min = self.mfr.get_data_as_array()[:, idx_min]
        values_max = self.mfr.get_data_as_array()[:, idx_max]
        array_ascending = self.mfr._return_by_score(rnd_score, ascending=True, return_type='array')
        self.assertTrue(np.all(array_ascending[:,0] == values_min) and np.all(array_ascending[:, -1] == values_max))

    def test_rank_by_univariate_f(self):
        shape = self.mfr.data.shape
        indices = pd.Index(['very_predictive', 'less_predictive'])
        self.mfr.data_columns = self.mfr.data_columns.append(indices)

        noise_small = np.random.normal(0, 1, shape[0])
        very_predictive = np.array(self.mfr.outcomes) + noise_small
        self.mfr.data = np.append(self.mfr.data, very_predictive[..., None], axis=1)

        noise_large = np.random.normal(0, 10, shape[0])
        less_predictive = np.array(self.mfr.outcomes) + noise_large
        self.mfr.data = np.append(self.mfr.data, less_predictive[..., None], axis=1)

        self.mfr.num_features=len(self.mfr.data_columns)
        names_descending = self.mfr.rank_by_univariate_f(ascending=False, return_type='names')
        self.assertTrue(names_descending[0]=='very_predictive')











