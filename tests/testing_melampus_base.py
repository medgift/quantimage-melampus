from melampus.melampus_base import Melampus
import tests.config as config
import pandas as pd
import numpy as np


data_df = pd.read_csv(config.path_to_test_data)
# create dummy outcome with categories as string
outcomes = config.gen_random_outcome(data_df, n_classes=2)
outcome_map = {0:'aaa', 1:'bbb'}
outcomes_string = [outcome_map[outcome] for outcome in outcomes]

# init melampus
mp = Melampus(dataframe=data_df, outcomes=outcomes_string)

# compare
mp.get_outcomes_as_array(categorical_as_codes=False) # -> mp.outcomes_labels
mp.get_outcomes_as_array(categorical_as_codes=True)  # -> mp.outcomes

# compare
mp = Melampus(dataframe=data_df, outcomes=outcomes_string)
mp.remove_nans(nan_policy='ignore')                  # only removes rows where outcome==nan

mp = Melampus(dataframe=data_df, outcomes=outcomes_string)
mp.remove_nans(nan_policy='drop-rows')               # removes rows there outcome or features == nan

mp = Melampus(dataframe=data_df, outcomes=outcomes_string)
mp.remove_nans(nan_policy='drop-columns')            # removes rows there outcome == nan and columns where features == nan
mp.get_data_as_dataframe()
mp.data_columns
# everything else as before.

from melampus.feature_ranking import MelampusFeatureRank
mfr = MelampusFeatureRank(dataframe=data_df, outcomes=outcomes_string)
mfr.remove_nans(nan_policy='drop-columns')
mfr.rank_by_univariate_f(ascending=False, return_type='names')