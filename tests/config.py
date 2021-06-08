from pathlib import Path
import numpy as np


def gen_random_outcome(df, n_classes=2):
    return list(np.random.randint(0, n_classes, df.shape[0]))

test_dir = Path(__file__).parent
path_to_test_data_features = test_dir.joinpath('data').joinpath('features_all.csv')
path_to_test_data_outcomes = test_dir.joinpath('data').joinpath('outcomes_all.csv')
path_to_test_data = test_dir.joinpath('data').joinpath('test_feature_file.csv')

test_output_dir   = test_dir.joinpath('output')
test_output_dir.mkdir(exist_ok=True, parents=True)