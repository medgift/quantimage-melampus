
from melampus.feature_selector import MelampusFeatureSelector
from melampus.preprocessor import MelampusPreprocessor
import pandas as pd
import numpy as np

path_to_data = "/home/daniel/repositories/quantimage-melampus/testing/features_album_Lymphangitis_Texture-Intensity-CT Optimized_CT-GTV L.csv"



def gen_random_outcome(df, n_classes=2):
    return list(np.random.randint(0, n_classes, df.shape[0]))

data = pd.read_csv(path_to_data)
outcomes = gen_random_outcome(data, n_classes=2)

mp = MelampusPreprocessor(filename=path_to_data, outcomes=outcomes)
#mfs = MelampusFeatureSelector(filename=path_to_data, outcomes=outcomes)