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
        self.assertRaises(Exception, Melampus, dataframe=self.data_df, target_col='outcomes_in_table')

    def test_init_access(self):
        mp = Melampus(filename=self.path_to_data, outcomes=self.outcomes)
        # check return types
        self.assertTrue(type(mp.get_data_as_array()==np.array))
        self.assertTrue(isinstance(mp.get_data_as_dataframe(),pd.DataFrame))
        # check if data is identical in both return scenarios
        self.assertTrue(np.all(mp.get_data_as_array()==mp.get_data_as_dataframe().values))
        # check if dataframe contains indices & outcomes
        df = mp.get_data_as_dataframe(include_outcomes=True, include_ids=True, outcome_name='outcome')
        self.assertIn('outcome', df.columns)
        for id_name in mp.id_names:
            self.assertIn(id_name, df.columns)






