
from melampus.melampus import Melampus
import tests.config as config
import pandas as pd
import numpy as np

path_to_feature_data = config.path_to_test_data
feature_data_df = pd.read_csv(path_to_feature_data)

# construct dummy outcome data (IDs: PatientID, Modality)
outcomes_df = feature_data_df[['PatientID', 'Modality']].drop_duplicates()
outcomes = config.gen_random_outcome(outcomes_df, n_classes=2)
outcomes_df['outcome_1'] = outcomes

# instantiate Melampus with feature data and matching outcome data
# you can provide either path or dataframe
M = Melampus(feature_data=feature_data_df, outcome_data=outcomes_df)

# M has maintains two instances of 'FeatureDB'
M.feature_db_orig # original data as read in from csv / provided as dataframe
M.feature_db_curr # copy of original data that is being updated
# M also has an instance of 'OutcomeDB' which keeps the outcome data
M.outcome_db

# Each of those '*DB's adds specific functionality to a common 'DB' class which in turn inherits from a 'dataIO' class.

# Most importantly, the DB class provides functionality
# 1) for defining IDs and handling the mapping between ids used internally and the column names provided in the csv/dataframe
# Internally, each DB instance maintains data as a pd.DataFrame
M.feature_db_orig.get_data_as_dataframe()
# and provides the functionality to extract the underlying data (removing the ID columns)
M.feature_db_orig.get_data_as_array()

# 2) '(un)merging' (stacking) IDs with column names, and flattening the resulting hierarchical column index into a single column with user-defined separators

# FeatureDB uses those latter functionalities to
# 1) merge ids into feature names
M.feature_db_orig.merge_ids_into_feature_names(id_names=['ROI'], level_separator_string='/')
M.feature_db_orig.get_data_as_dataframe()
# 2) and to extract them again, i.e. undo the previous step
M.feature_db_orig.extract_ids_from_feature_names(id_names=['ROI'], level_separator_string='/')
M.feature_db_orig.get_data_as_dataframe()

# Melampus compares the IDs provided which the outcome data (OutcomeDB) to the IDs of the feature data to reshape the
# feature data in such a way that the IDs provided by the outcome data are unique.

# In the example here the Feature data has IDs [PatientID, Modality, ROI] and the outcome data has IDs [PatientId, Modality]
# To uniquely match outcomes and feature data, a new 'feature' has to be created for each ROI

# Melampus does this during initialization, and maintains the transformed FeatureDB instance as 'current' db used for further processing
M.feature_db_curr.get_data_as_dataframe() # has IDs ['patient_id', 'modality']

# feel free to play with this, e.g. providing outcome data with only a single ID


# Notes:
# 1) I realized that DB.get_data_as_dataframe() returns the internal identifiers as column names, not the original ones
#    provided with the data. I will change this so that user interaction is entirely based on the original IDs
# 2) DB relies on a separator string to encode and decode composite column names. 
#    Melampus currently uses '/' as default separator; I would usually use '_' for this purpose, but feature names are 
#    already using this. 
#    We could define a similar mechanism for grouping/ungrouping feature (sub)classes, however, I don't know if the 
#    current naming is sufficiently consistent. Do all feature names have the same number of 'sub categories'?
