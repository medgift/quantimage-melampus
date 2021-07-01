from melampus.database import FeatureDB, OutcomeDB
import pandas as pd
from sklearn.feature_selection import f_classif
import numpy as np
import copy

class Melampus(object):

    def __init__(self, feature_data = None, outcome_data = None,
                 id_map = {'patient_id': 'PatientID', 'modality': 'Modality', 'roi': 'ROI'},
                 flattening_separator = '/'):

        self.id_map = id_map
        self.flattening_separator = flattening_separator

        if feature_data is not None:
            self.add_feature_data(feature_data)
        if outcome_data is not None:
            self.add_outcome_data(outcome_data)

        self._check_reset_data()
        self._match_index_levels_features_to_outcomes()
        self._align_samples_features_outcomes()
        self._check_consistency_features_outcomes()

    def reset_current(self):
        self._check_reset_data()
        self._match_index_levels_features_to_outcomes()
        self._align_samples_features_outcomes()
        self._check_consistency_features_outcomes()

    def add_feature_data(self, data):
        if not hasattr(self, 'feature_db_orig'):
            self.feature_db_orig = FeatureDB(data=data, id_int_to_ext_map=self.id_map,
                                             flattening_separator=self.flattening_separator)
        else:
            raise Exception("Feature data already defined")

    def add_outcome_data(self, data):
        if not hasattr(self, 'outcome_db_orig'):
            self.outcome_db_orig = OutcomeDB(data=data, id_int_to_ext_map=self.id_map,
                                             flattening_separator=self.flattening_separator)
        else:
            raise Exception("Outcome data already defined")

    def _match_index_levels_features_to_outcomes(self):
        ids_outcomes = self.outcome_db_curr._get_level_names(which='index')
        ids_features = self.feature_db_curr._get_level_names(which='index')
        feature_ids_to_flatten = [self.id_map[id] for id in ids_features if id not in ids_outcomes]
        #print('feature ids to flatten ',feature_ids_to_flatten)
        self.feature_db_curr.merge_ids_into_feature_names(id_names=feature_ids_to_flatten)
        self.feature_db_curr.dataframe.sort_index(inplace=True)
        self.outcome_db_curr.dataframe.sort_index(inplace=True)

    def _align_samples_features_outcomes(self):
        idx_features = self.feature_db_curr.get_data_as_dataframe(orig_ids=True).index
        idx_outcomes = self.outcome_db_curr.get_data_as_dataframe(orig_ids=True).index
        sample_idx_in_features_not_outcomes = idx_features.difference(idx_outcomes)
        sample_idx_in_outcomes_not_features = idx_outcomes.difference(idx_features)
        if len(sample_idx_in_features_not_outcomes)>0:
            print("Sample IDs in features, not outcomes: %i"%len(sample_idx_in_features_not_outcomes))
            print(sample_idx_in_features_not_outcomes)
        if len(sample_idx_in_outcomes_not_features)>0:
            print("Sample IDs in outcomes, not features: %i"%len(sample_idx_in_outcomes_not_features))
            print(sample_idx_in_outcomes_not_features)
        self.feature_db_curr.dataframe.drop(sample_idx_in_features_not_outcomes, inplace=True)
        self.outcome_db_curr.dataframe.drop(sample_idx_in_outcomes_not_features, inplace=True)

    def _check_consistency_features_outcomes(self):
        # check number of entries
        n_entries_features = len(self.feature_db_curr.dataframe)
        n_entries_outcomes = len(self.outcome_db_curr.dataframe)
        if n_entries_features!=n_entries_outcomes:
            raise Exception("Feature DB has %i entries but Outcome DB has %i"%(n_entries_features, n_entries_outcomes))
        # check indices (after sorting)
        self.feature_db_curr.dataframe.sort_index(inplace=True)
        self.outcome_db_curr.dataframe.sort_index(inplace=True)
        idx_features = self.feature_db_curr.dataframe.index
        idx_outcomes = self.outcome_db_curr.dataframe.index
        if not idx_features.equals(idx_outcomes):
            raise Exception("Indices of feature and outcome DBs differ")

    def _check_reset_data(self):
        if hasattr(self, 'outcome_db_orig') and hasattr(self, 'feature_db_orig'):
            print("Resetting outcome and feature DBs: orig -> current")
            self.feature_db_curr = copy.deepcopy(self.feature_db_orig)
            self.outcome_db_curr = copy.deepcopy(self.outcome_db_orig)
        else:
            print("Outcome and/or Feature data is still missing")

    def get_feature_data_as_array(self, current=True, **kwargs):
        if current:
            return  self.feature_db_curr.get_data_as_array(**kwargs)
        else:
            return  self.feature_db_orig.get_data_as_array(**kwargs)

    def get_feature_data_as_dataframe(self, current=True,**kwargs):
        if current:
            return  self.feature_db_curr.get_data_as_dataframe(**kwargs)
        else:
            return  self.feature_db_orig.get_data_as_dataframe(**kwargs)

    def get_outcome_data_as_array(self, current=True, **kwargs):
        if current:
            return  self.outcome_db_curr.get_data_as_array(**kwargs)
        else:
            return  self.outcome_db_orig.get_data_as_array(**kwargs)


    def get_outcome_data_as_dataframe(self, current=True, **kwargs):
        if current:
            return  self.outcome_db_curr.get_data_as_dataframe(**kwargs)
        else:
            return  self.outcome_db_orig.get_data_as_dataframe(**kwargs)

    def get_data_as_dataframe(self):
        merged_df = pd.merge(self.get_feature_data_as_dataframe(), self.get_outcome_data_as_dataframe(), left_index=True,
                             right_index=True, how="left")
        return merged_df

    def _update_feature_data(self, data_array):
        self.feature_db_curr._update_data(data_array)

    def _update_outcome_data(self, data_array):
        self.outcome_db_curr._update_data(data_array)

    def _get_db(self, col_name, current=True):
        if current:
            db_features = self.feature_db_curr
            db_outcomes = self.outcome_db_curr
        else:
            db_features = self.feature_db_orig
            db_outcomes = self.outcome_db_orig
        if col_name in db_features.get_feature_names():
            return db_features
        elif col_name in db_outcomes.get_outcome_names():
            return db_outcomes
        else:
            raise Exception("Could not locate column '%s'"%col_name)

    def assert_type_datetime(self, name, format="%Y-%m-%d", orig_db=True):
        db = self._get_db(name, current=not orig_db)
        db.assert_type_datetime(name=name, format=format)

    def assert_type_categorical(self, name, order=None, orig_db=True):
        db = self._get_db(name, current= not orig_db)
        db.assert_type_categorical(name=name, order=order)

    def select_on_levels(self, selection_dict: dict):
        self._check_reset_data()

        ids_outcomes = self.outcome_db_orig._get_level_names(which='index', id_as_external=True)
        selection_dict_outcomes = {key: values for key, values in selection_dict.items() if key in ids_outcomes}
        self.outcome_db_curr.select_on_levels(selection_dict_outcomes, inplace=True)

        ids_features = self.feature_db_orig._get_level_names(which='index', id_as_external=True)
        selection_dict_features = {key: values for key, values in selection_dict.items() if key in ids_features}
        self.feature_db_curr.select_on_levels(selection_dict_features, inplace=True)

        self._match_index_levels_features_to_outcomes()
        self._check_consistency_features_outcomes()

    def select_features(self, features_to_keep=None, features_to_remove=None, by='name', inplace=True):
        features_avail = self.feature_db_curr.dataframe.columns
        if by=='index': # convert available features to index ids
            features_avail = list(enumerate(features_avail))
        if features_to_keep is None:
            features_to_keep = features_avail
        if features_to_remove is None:
            features_to_remove = []
        features_sel = [f for f in features_to_keep if (not f in features_to_remove)]
        print("Selected features: ", features_sel)
        if by=='index': # convert selected index ids to feature names
            features_sel = self.map_feature_index_to_name(features_sel)
        df_features = self.feature_db_curr.dataframe[features_sel]
        if inplace:
            self.feature_db_curr.dataframe = df_features
        else:
            return df_features

    def select_outcome(self, target_outcome, inplace=True):
        if target_outcome in self.get_outcome_data_as_dataframe().columns:
            df_outcome = self.outcome_db_curr.dataframe[[target_outcome]]
            if inplace:
                self.outcome_db_curr.dataframe = df_outcome
            else:
                return df_outcome
        else:
            raise Exception("Outcome '%s' is not available"%target_outcome)

    def remove_nans(self, nan_policy='ignore', inplace=True):
        """

        :param nan_policy:
            - 'ignore': drops rows where outcome == NaN
            - 'drop-rows': drops rows where outcome == NaN and at least 1 feature value == Nan
            - 'drop-columns': drops rows where outcome == NaN and columns where at least 1 feature value == Nan
        :param inplace:
        :return:
        """
        outcome_cols = self.get_outcome_data_as_dataframe().columns
        if len(outcome_cols) >1 :
            raise Exception("Select outcome first")
        else:
            outcome_series = self.outcome_db_curr.dataframe[outcome_cols[0]]
            sel_outcome_not_nan = ~outcome_series.isna()

            df_features = self.feature_db_curr.dataframe
            sel_features_not_nan_by_row = (df_features.isna().sum(axis=1) == 0)
            sel_features_not_nan_by_col = (df_features.isna().sum(axis=0) == 0)

            # always drop rows where outcome is NaN
            if nan_policy=='drop-columns':
                df_outcome = self.outcome_db_curr.dataframe.loc[sel_outcome_not_nan]
            elif nan_policy=='drop-rows':
                df_outcome = self.outcome_db_curr.dataframe.loc[sel_features_not_nan_by_row & sel_outcome_not_nan]
            elif nan_policy == 'ignore':
                df_outcome = self.outcome_db_curr.dataframe.loc[sel_outcome_not_nan]

            if inplace:
                self.outcome_db_curr.dataframe = df_outcome

            # for features, drop either columns or rows
            if nan_policy=='drop-columns':
                df_features = df_features.loc[sel_outcome_not_nan].dropna(axis=1)
            elif nan_policy=='drop-rows':
                df_features = df_features.loc[sel_features_not_nan_by_row & sel_outcome_not_nan]
            elif nan_policy=='ignore':
                df_features = df_features.loc[sel_outcome_not_nan]
            if inplace:
                self.feature_db_curr.dataframe = df_features

            return df_features, df_outcome

    def get_data_for_analysis(self, return_as='array', return_category_code=True):
        self._check_consistency_features_outcomes()
        if return_as=='array':
            X_features = self.get_feature_data_as_array(return_category_code=return_category_code)
            y_outcome  = self.get_outcome_data_as_array(return_category_code=return_category_code)
        elif return_as=='dataframe':
            X_features = self.get_feature_data_as_dataframe(return_category_code=return_category_code)
            y_outcome = self.get_outcome_data_as_dataframe(return_category_code=return_category_code)
        return X_features, y_outcome

    def map_feature_index_to_name(self, index_list):
        return self.feature_db_curr.get_feature_names()[index_list]



