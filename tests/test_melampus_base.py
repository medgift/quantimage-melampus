import unittest
from melampus.melampus_base import Melampus
import tests.config as config
import pandas as pd
import numpy as np


class TestMelampus(unittest.TestCase):

    def setUp(self):
        self.path_to_data = config.path_to_test_data
        self.data_df = pd.read_csv(self.path_to_data)
        self.outcomes = config.gen_random_outcome(self.data_df, n_classes=2)

    def test_init_exceptions(self):
        # expected to raise exception -> target_col / outcomes missing
        self.assertRaises(Exception, Melampus, filename=self.path_to_data)
        # expected to raise exception -> only one of filename / dataframe
        self.assertRaises(Exception, Melampus, filename=self.path_to_data, dataframe=self.data_df, outcomes=self.outcomes)
        # expected to raise exception -> only one of target_col / outcomes
        data_df = self.data_df.copy()
        data_df['outcomes_in_table'] = self.outcomes
        self.assertRaises(Exception, Melampus, dataframe=data_df, target_col='outcomes_in_table', outcomes=self.outcomes)
        # expected to raise exception -> target_col not in dataframe
        print(self.data_df.columns)
        self.assertRaises(Exception, Melampus, dataframe=self.data_df, target_col='outcomes_in_table')

    def test_init_access(self):
        mp = Melampus(filename=self.path_to_data, outcomes=self.outcomes)
        # check return types
        self.assertTrue(type(mp.get_data_as_array()==np.array))
        self.assertTrue(isinstance(mp.get_data_as_dataframe(),pd.DataFrame))
        # check if data is identical in both return scenarios
        diff = mp.get_data_as_array()-mp.get_data_as_dataframe().values
        self.assertTrue(np.nansum(mp.get_data_as_array()-mp.get_data_as_dataframe().values)<1E-6)
        # check if dataframe contains indices & outcomes
        df = mp.get_data_as_dataframe(include_outcomes=True, include_ids=True, outcome_name='outcome')
        self.assertIn('outcome', df.columns)
        for id_name in mp.id_column_names:
            self.assertIn(id_name, df.columns)

    def test_string_outcome(self):
        outcome_map = {0: 'aaa', 1: 'bbb'}
        outcomes_string = [outcome_map[outcome] for outcome in self.outcomes]
        mp = Melampus(dataframe=self.data_df, outcomes=outcomes_string)
        self.assertTrue(hasattr(mp, 'outcomes_series'))
        self.assertTrue(hasattr(mp, 'outcomes'))
        self.assertTrue(hasattr(mp, 'outcomes_labels'))

        self.assertTrue(mp.outcomes_series.dtype.name=='category')
        self.assertTrue(isinstance(mp.outcomes[0], int))
        self.assertTrue(isinstance(mp.outcomes_labels[0], str))

        print(mp.get_data_as_dataframe(categorical_as_codes=False, include_outcomes=True, outcome_name='outcome')['outcome'])
        self.assertTrue(mp.get_data_as_dataframe(categorical_as_codes=False, include_outcomes=True, outcome_name='outcome')['outcome'].dtype.name=='object')
        self.assertTrue(mp.get_data_as_dataframe(categorical_as_codes=True, include_outcomes=True, outcome_name='outcome')['outcome'].dtype.name.startswith('int'))

    def test_remove_nan(self):
        #self.data_df[10,'log-sigma-2-mm-3D_firstorder_Kurtosis'] = np.nan
        self.outcomes[5] = np.nan
        #-- ignore -> remove column where outcome = NaN
        mp1 = Melampus(dataframe=self.data_df, outcomes=self.outcomes)
        n_cols_nan = np.sum(mp1.get_data_as_dataframe().isna().sum()>0)
        n_rows_nan = np.sum(mp1.get_data_as_dataframe().isna().sum(axis=1) > 0)
        print(n_rows_nan)
        self.assertTrue(mp1.get_data_as_dataframe().isna().sum().sum()>0)
        shape_nan = mp1.get_data_as_dataframe().shape
        mp1.remove_nans(nan_policy='ignore')
        self.assertListEqual(list(shape_nan), list(np.array(mp1.get_data_as_dataframe().shape) + np.array([1,0])))
        self.assertEqual(shape_nan[0], len(mp1.get_outcomes_as_array()) + 1)
        #-- drop-column -> remove column features = nan
        mp2 = Melampus(dataframe=self.data_df, outcomes=self.outcomes)
        mp2.remove_nans(nan_policy='drop-columns')
        self.assertListEqual(list(shape_nan), list(np.array(mp2.get_data_as_dataframe().shape) + np.array([1,n_cols_nan])))
        self.assertEqual(shape_nan[0], len(mp2.get_outcomes_as_array()) + 1)
        #-- drop-rows -> remove column features = nan
        mp3 = Melampus(dataframe=self.data_df, outcomes=self.outcomes)
        mp3.remove_nans(nan_policy='drop-rows')
        self.assertListEqual(list(shape_nan), list(np.array(mp3.get_data_as_dataframe().shape) + np.array([n_rows_nan,0])))
        self.assertEqual(shape_nan[0], len(mp3.get_outcomes_as_array()) + n_rows_nan)

