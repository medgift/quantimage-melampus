from melampus.database import FeatureDB, OutcomeDB


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

        self.process_check_data()

    def add_feature_data(self, data):
        if not hasattr(self, 'feature_db_orig'):
            self.feature_db_orig = FeatureDB(data=data, id_int_to_ext_map=self.id_map,
                                             flattening_separator=self.flattening_separator)
            self.feature_db_curr = FeatureDB(data=data, id_int_to_ext_map=self.id_map,
                                             flattening_separator=self.flattening_separator)
        else:
            raise Exception("Feature data already defined")

    def add_outcome_data(self, data):
        if not hasattr(self, 'outcome_db'):
            self.outcome_db = OutcomeDB(data=data, id_int_to_ext_map=self.id_map,
                                        flattening_separator=self.flattening_separator)
        else:
            raise Exception("Outcome data already defined")

    def _match_index_levels_features_to_outcomes(self):
        ids_outcomes = self.outcome_db._get_level_names(which='index')
        ids_features = self.feature_db_curr._get_level_names(which='index')
        feature_ids_to_flatten = [self.id_map[id] for id in ids_features if id not in ids_outcomes]
        self.feature_db_curr.merge_ids_into_feature_names(id_names=feature_ids_to_flatten)

    def _merge_features_outcomes(self):
        # 'join' is left join
        merged_df = self.feature_db_curr.dataframe.join(self.outcome_db.dataframe)

    def _check_consistency_features_outcomes(self):
        n_entries_features = len(self.feature_db_curr.dataframe)
        n_entries_outcomes = len(self.outcome_db.dataframe)
        if n_entries_features!=n_entries_outcomes:
            raise Exception("Feature DB has %i entries but Outcome DB has %i"%(n_entries_features, n_entries_outcomes))
        else:
            print("Successfully initialized feature and outcome data -- Ready to go!")

    def process_check_data(self):
        if hasattr(self, 'outcome_db') and hasattr(self, 'feature_db_curr'):
            self._match_index_levels_features_to_outcomes()
            self._check_consistency_features_outcomes()
        else:
            print("Outcome and/or Feature data is still missing")

    def _get_feature_data(self, current=True):
        if current:
            return  self.feature_db_curr.get_data_as_array
        else:
            return  self.feature_db_orig.get_data_as_array

    def _update_feature_data(self, data_array):
        self.feature_db_curr._update_data(data_array)
