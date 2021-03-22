from pandas import DataFrame, read_csv
from pathlib import Path

class Melampus(object):
    """
    This is the Melampus base class from which further Melampus Classes inherit.
    It provides basic functionality for reading and managing feature data.
    Melampus accepts a csv file or pandas.DataFrame object with **column names included**.
    The name of the target variable can also be given as an optional parameter in case that the target variable is
    included in the csv file or pandas.DataFrame.

    :param filename: The name of the csv file that includes data and identifiers
    :param dataframe: A pandas.DataFrame object that includes data and identifiers
    :param target_col: Name of the target variable if included in the csv dataset or pandas.DataFrame, defaults to None
    :param outcomes: List of outcome values, length must equal the number of cases in the dataset
    :param id_names_map: Dictionary that maps standard identifier names ('patient_id', 'modality', 'roi') to those
    used in the data.
    The transformed datasets are stored and can be accessed on self.data object in numpy array format,
    or (including ids and outcome) as pandas.DataFrame on self.get_dataframe().
    """

    def __init__(self, filename: str = None, dataframe: DataFrame = None, target_col=None, outcomes=[],
                 id_names_map = {'patient_id' : 'PatientID', 'modality' : 'Modality', 'roi' : 'ROI'}):

        self.id_names_map = id_names_map
        self.id_names = list(id_names_map.keys())
        self.id_column_names = list(id_names_map.values())

        # check for filename vs dataframe and process
        self.filename = filename
        if self.filename and not isinstance(dataframe, DataFrame):
            dataframe  = self._read_csv(self.filename)
        elif isinstance(dataframe, DataFrame) and not self.filename:
            pass
        else:
            raise Exception("Specify one of 'filename' / 'dataframe'")

        self.ids, data_tmp = self._split_columns_from_data(dataframe, self.id_column_names)

        # check for target_col vs outcomes and process
        self.target_col = target_col
        if self.target_col is not None and len(outcomes)==0:
            outcomes, data_tmp = self._split_columns_from_data(data_tmp, self.target_col)
            self.outcomes = list(outcomes.values)
        elif self.target_col is None and len(outcomes)>0:
            if len(outcomes) != len(data_tmp):
                raise Exception("Inconsistent number of cases between outcomes (%i) and samples (%i)"%(
                                                                                len(self.outcomes, self.num_cases)))
            else:
                self.outcomes = outcomes
        else:
            raise Exception("Specify one of 'target_col' / 'outcomes'")

        # assign cleaned data table
        self.data_columns = data_tmp.columns # pd.Index
        self.data = data_tmp.values  # self.data is updated by transformations and represents the current 'state'

        # compute num_cases / per class
        self.num_features = len(self.data_columns)
        self.num_cases = len(self.ids)
        self.num_cases_in_each_class = {}
        for el in self.outcomes:
            self.num_cases_in_each_class[el] = self.outcomes.count(el)

    def _read_csv(self, filename: str = None):
        if not filename:
            filename = self.filename
        try:
            return read_csv(filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(e)

    def _split_columns_from_data(self, dataframe: DataFrame, column_names: list):
        data_sel = None
        data_remaining = dataframe
        try:
            data_sel = dataframe[column_names]
            data_remaining = dataframe.drop(columns=column_names)
        except KeyError as e:
            print(e)
        return data_sel, data_remaining

    def get_data_as_array(self):
        return self.data

    def get_data_as_dataframe(self, include_ids=False, include_outcomes=False, outcome_name='outcome'):
        df = DataFrame(columns=self.data_columns, data=self.data)
        if include_ids:
            df[self.id_column_names] = self.ids
        if include_outcomes:
            df[outcome_name] = self.outcomes
        return df

    def export_data(self, path_to_file, **kwargs):
        path_to_file = Path(path_to_file)
        path_to_file.parent.mkdir(exist_ok=True, parents=True)
        ext = path_to_file.name.split('.')[-1]
        df = self.get_data_as_dataframe(**kwargs)
        if ext=='csv':
            df.to_csv(path_to_file)
        elif exit=='xlsx':
            df.to_excel(path_to_file)
        elif exit=='pkl':
            df.to_pickle(path_to_file)
        else:
            raise Exception("Export to file extension '%s' not implemented"%ext)

