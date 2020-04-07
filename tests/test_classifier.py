import unittest
from melampus.preprocessor import Preprocessor


class TestClassifier(unittest.TestCase):
    def test_clf_fit_train_data_exists_should_not_pass(self):
        pass
        #self.assertRaises(FileNotFoundError, Preprocessor, 'synt')

