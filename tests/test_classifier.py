import unittest
from melampus.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):
    def test_read_file(self):
        self.assertRaises(FileNotFoundError, Preprocessor, 'synt')
        self.assertRaises(KeyError, Preprocessor, 'tests/mockfiles/output_L0_processed.csv')
        with self.assertRaises(Exception):
            Preprocessor('tests/mockfiles/output_L0_GTVL.csv', target_col = 'NAME_NOT_EXISTS')

