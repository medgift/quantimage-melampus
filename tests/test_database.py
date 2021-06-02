import unittest
from melampus.database import DB, FeatureDB, OutcomeDB
import tests.config as config
import pandas as pd
import numpy as np


class TestDataBase(unittest.TestCase):

    def setUp(self):
        self.path_to_data = config.path_to_test_data
        self.data_df = pd.read_csv(self.path_to_data)
        self.outcomes = config.gen_random_outcome(self.data_df, n_classes=2)
        self.db = DB(data=self.path_to_data,
                     id_int_to_ext_map = {'patient_id' : 'PatientID', 'modality' : 'Modality', 'roi' : 'ROI'},
                     flattening_separator='/')

    def test_init(self):
        # expected to raise exception -> mapping roi2 does not exist
        # Changed code so that only id names available in DF are considered
        #self.assertRaises(Exception, DB, data=self.path_to_data, id_int_to_ext_map = {'patient_id' : 'PatientID', 'modality' : 'Modality', 'roi2' : 'ROI2'})
        # check mapping dicts
        self.assertDictEqual(self.db.id_names_map_int_to_ext, {'patient_id' : 'PatientID', 'modality' : 'Modality', 'roi' : 'ROI'})
        self.assertDictEqual(self.db.id_names_map_ext_to_int, {'PatientID' : 'patient_id', 'Modality' : 'modality', 'ROI' : 'roi'})
        # check indices dataframe
        self.assertListEqual(self.db.dataframe.index.names, ['patient_id', 'modality', 'roi'])
        # check index status
        self.assertEqual(self.db.df_index_status,'internal')

    def test_map_dataframe_id_names(self):
        # indices internal -> external
        self.db._map_dataframe_index_to_original(inplace=True)
        self.assertListEqual(self.db.dataframe.index.names, ['PatientID', 'Modality', 'ROI'])
        self.assertEqual(self.db.df_index_status, 'original')
        # indices external -> internal
        self.db._map_dataframe_index_to_internal(inplace=True)
        self.assertListEqual(self.db.dataframe.index.names, ['patient_id', 'modality', 'roi'])
        self.assertEqual(self.db.df_index_status, 'internal')

    def test_merge_ids_with_columns(self):
        # expected to raise exception -> mapping roi2 does not exist
        self.assertRaises(Exception, self.db.merge_ids_with_columns, id_names=['roi2'])
        # merge ROI into column names
        self.db.merge_ids_with_columns(id_names=['ROI'])
        self.assertListEqual(self.db.dataframe.index.names, ['patient_id', 'modality'])
        # raise exception if trying to merge 'ROI' again
        self.assertRaises(Exception, self.db.merge_ids_with_columns, id_names=['ROI'])

    def test_unmerge_ids_from_columns(self):
        # merge ROI into column names
        self.db.merge_ids_with_columns(id_names=['ROI'])
        self.assertListEqual(self.db.dataframe.index.names, ['patient_id', 'modality'])
        # expected to raise exception -> mapping roi does not exist, require external names
        self.assertRaises(Exception, self.db.unmerge_ids_from_columns, id_names=['roi'])
        self.db.unmerge_ids_from_columns(id_names=['ROI'])
        self.assertListEqual(self.db.dataframe.index.names, ['patient_id', 'modality', 'roi'])
        # raise exception if trying to unmerge 'ROI' again
        self.assertRaises(Exception, self.db.unmerge_ids_from_columns, id_names=['ROI'])

    def test_flatten_column_index(self):
        self.assertRaises(Exception, self.db.flatten_column_index)
        self.db.merge_ids_with_columns(id_names=['ROI'])
        self.assertTrue(isinstance(self.db.dataframe.columns, pd.MultiIndex))
        self.db.flatten_column_index()
        self.assertFalse(isinstance(self.db.dataframe.columns, pd.MultiIndex))

    def test_unflatten_column_index(self):
#        self.assertRaises(Exception, self.db.unflatten_column_index)
        self.db.merge_ids_with_columns(id_names=['ROI'])
        self.db.flatten_column_index()
        self.assertFalse(isinstance(self.db.dataframe.columns, pd.MultiIndex))
        self.assertRaises(Exception, self.db.unflatten_column_index, level_names='roi')
        self.db.unflatten_column_index()
        self.assertTrue(isinstance(self.db.dataframe.columns, pd.MultiIndex))

    def test_get_data_as_array(self):
        self.assertTrue(np.array_equal(self.db.dataframe.values, self.db.get_data_as_array(), equal_nan=True))

    def test_get_data_as_dataframe(self):
        self.assertTrue(np.array_equal(self.db.dataframe.values, self.db.get_data_as_dataframe().values, equal_nan=True))

    def test_update_data(self):
        new_data = np.random.random(self.db.get_data_as_array().shape)
        self.db._update_data(new_data)
        self.assertTrue(np.array_equal(self.db.get_data_as_array(), new_data))
        new_data2 = np.random.random(np.array(self.db.get_data_as_array().shape) + np.array([1,1]))
        self.assertRaises(Exception, self.db._update_data, new_data2)

    def test_get_default_levels(self):
        self.assertListEqual(self.db._get_default_levels(mode='merge'), ['Modality', 'ROI'])
        self.assertListEqual(self.db._get_default_levels(mode='unmerge'), [])
        self.db.merge_ids_with_columns(id_names=['ROI'])
        self.db.flatten_column_index()
        self.assertEqual(self.db._get_default_levels(mode='merge'), 'Modality')
        self.assertEqual(self.db._get_default_levels(mode='unmerge'), 'ROI')



class TestFeatureDataBase(unittest.TestCase):

    def setUp(self):
        self.path_to_data = config.path_to_test_data
        self.data_df = pd.read_csv(self.path_to_data)
        self.outcomes = config.gen_random_outcome(self.data_df, n_classes=2)
        self.db = FeatureDB(data=self.path_to_data,
                     id_int_to_ext_map = {'patient_id' : 'PatientID', 'modality' : 'Modality', 'roi' : 'ROI'},
                     flattening_separator='/')

    def test_merge_ids_into_feature_names(self):
        levels_roi = self.db.get_data_as_dataframe().index.levels[2]
        levels_modality = self.db.get_data_as_dataframe().index.levels[1]
        n_levels_roi = len(levels_roi)
        n_levels_modality = len(levels_modality)
        n_features = self.db.get_n_features()
        self.db.merge_ids_into_feature_names(id_names=['ROI'])
        self.assertEqual(n_levels_roi * n_features, self.db.get_n_features())

        self.db.merge_ids_into_feature_names(id_names=['Modality'])
        self.assertEqual(n_levels_roi * n_levels_modality * n_features, self.db.get_n_features())

        self.db.extract_ids_from_feature_names()
        self.assertEqual(n_features, self.db.get_n_features())




class TestOutcomeDataBase(unittest.TestCase):

    def setUp(self):
        self.path_to_data = config.path_to_test_data
        self.data_df = pd.read_csv(self.path_to_data)
        outcomes_df = self.data_df.copy()[['PatientID', 'Modality']].drop_duplicates()
        outcomes_df['outcome_1'] = config.gen_random_outcome(outcomes_df, n_classes=2)
        self.outcomes_df = outcomes_df
        self.db = OutcomeDB(data=self.outcomes_df,
                     id_int_to_ext_map = {'patient_id' : 'PatientID', 'modality' : 'Modality', 'roi' : 'ROI'},
                     flattening_separator='/')

    def test_assert_outcome_categorical(self):
        self.db._assert_categorical(outcome_name='outcome_1')
        self.assertTrue(self.db.dataframe.outcome_1.dtype.name== 'category')

    def test_expand_categorical(self):
        self.db._expand_categorical(outcome_name='outcome_1')
        self.assertTrue(self.db.dataframe.outcome_1.dtype.name== 'category')
        self.assertTrue(self.db.dataframe.outcome_1_codes.dtype.name == 'int8')








