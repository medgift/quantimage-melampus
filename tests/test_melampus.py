import unittest
from melampus.melampus import Melampus
import tests.config as config
import pandas as pd
import numpy as np


class TestMelampus(unittest.TestCase):

    def setUp(self):
        self.path_to_data = config.path_to_test_data
        self.data_df = pd.read_csv(self.path_to_data)
        # Construct outcome data with one outcome per patientId-roi
        outcomes_df = self.data_df.copy()[['PatientID', 'ROI']].drop_duplicates()
        outcomes_df['outcome_1'] = config.gen_random_outcome(outcomes_df, n_classes=2)
        self.outcomes_df = outcomes_df
        self.melampus = Melampus(feature_data=self.data_df, outcome_data=self.outcomes_df)
        print(self.melampus.get_data_as_dataframe())

    def test_select_on_levels(self):
        self.melampus.select_on_levels({'ROI':'GTV L'})
        roi_levels_after  = list(self.melampus.get_data_as_dataframe().index.levels[1].unique())
        self.assertListEqual(roi_levels_after, ['GTV L'])