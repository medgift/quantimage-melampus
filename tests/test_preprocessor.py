import unittest
from melampus.preprocessor import MelampusPreprocessor


class TestPreprocessor(unittest.TestCase):
    def test_read_file(self):
        self.assertRaises(FileNotFoundError, MelampusPreprocessor, 'synt')
        self.assertRaises(KeyError, MelampusPreprocessor, 'tests/mockfiles/output_L0_processed.csv')
        with self.assertRaises(Exception):
            MelampusPreprocessor('tests/mockfiles/output_L0_GTVL.csv', target_col='NAME_NOT_EXISTS')

