import unittest
from melampus.classifier import MelampusClassifier


class TestClassifier(unittest.TestCase):
    def test_clf_fit_train_data_exists_should_not_pass(self):
        outcomes = []
        mel_clf = MelampusClassifier(filename='synthetic_data/output_L0_GTVL.csv', algorithm_name='elastic_net',
                                     outcomes=outcomes, normalize=True, scaling=True, dim_red=(True, 5))
        self.assertRaises(ValueError, mel_clf.train)


