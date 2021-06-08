from melampus.classifier import MelampusClassifier
import pandas as pd
import tests.config as config




path_to_data = config.path_to_test_data
data_df = pd.read_csv(path_to_data)
data_sel = data_df[data_df.Modality=='CT']
cols_to_drop = [name for name in data_sel.columns if name.startswith('PET')]
data_sel.drop(columns=['PatientID']+cols_to_drop, inplace=True)
outcomes = config.gen_random_outcome(data_sel, n_classes=2)
classifier = MelampusClassifier(dataframe=data_sel, algorithm_name='elastic_net',
                                outcomes=outcomes, normalize=True, scaling=True, dim_red=(True, 5))


result = classifier.train_and_evaluate
print(result)

