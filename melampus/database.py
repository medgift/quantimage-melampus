from pandas import DataFrame, MultiIndex, read_csv
from pathlib import Path


class DataIO(object):
    def __init__(self, data):

        if isinstance(data, str) or isinstance(data, Path):
            self.path_to_file = data
            dataframe = self._read_csv()
        elif isinstance(data, DataFrame):
            dataframe = data
        else:
            raise Exception("Attribute 'data' cannot be of type '%s'; expect str or pd.DataFrame"%type(data))

        self.dataframe = dataframe.copy()

    def _read_csv(self, path_to_file=None):
        if path_to_file is None:
            path_to_file = self.path_to_file
        try:
            return read_csv(path_to_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(e)

    def export_dataframe(self, path_to_file):
        path_to_file = Path(path_to_file)
        path_to_file.parent.mkdir(exist_ok=True, parents=True)
        ext = path_to_file.name.split('.')[-1]
        if ext == 'csv':
            self.dataframe.to_csv(path_to_file)
        elif exit == 'xlsx':
            self.dataframe.to_excel(path_to_file)
        elif exit == 'pkl':
            self.dataframe.to_pickle(path_to_file)
        else:
            raise Exception("Export to file extension '%s' not implemented" % ext)


class DB(DataIO):
    def __init__(self, data,
                 id_int_to_ext_map = {'patient_id' : 'PatientID', 'modality' : 'Modality', 'roi' : 'ROI'},
                 data_name = 'data',
                 flattening_separator = '_'):

        # read / initialize data -> self.dataframe
        super().__init__(data)

        # initialize attributes used later
        self.flattened_column_levels = []
        self.flattening_separator = flattening_separator
        self.data_name = data_name

        self._update_ids(id_int_to_ext_map=id_int_to_ext_map)

    def _update_ids(self, id_int_to_ext_map):
        # set ids
        self.id_names_map_int_to_ext = id_int_to_ext_map
        self.id_names_map_ext_to_int = {value: key for key, value in self.id_names_map_int_to_ext.items()}
        # print(self.id_names_map_ext_to_int)

        # rename ID columns & set index
        self.dataframe.columns.names = [self.data_name]
        self._map_dataframe_id_names(external_to_internal=True)

    def _map_dataframe_id_names(self, external_to_internal=True, inplace=True):
        if external_to_internal:
            map = self.id_names_map_ext_to_int
        else:
            map = self.id_names_map_int_to_ext
        if inplace:
            self.dataframe.reset_index(drop=True, inplace=True)
            self.dataframe.rename(columns=map, inplace=True)
            if len(map)>0:
                self.dataframe.set_index(list(map.values()), inplace=True)
        else:
            df = self.dataframe.reset_index(drop=False, inplace=False)
            df.rename(columns=map, inplace=True)
            print(df)
            if len(map)>0:
                df.set_index(list(map.values()), inplace=True)
            return df

    def export_dataframe(self, path_to_file, orig_ids=True):
        if orig_ids:
            self._map_dataframe_id_names(external_to_internal=False)
        super().export_dataframe(path_to_file)
        if orig_ids:
            self._map_dataframe_id_names(external_to_internal=True)

    def merge_ids_with_columns(self, id_names=[]):
        for id_name_ext in id_names:
            if id_name_ext in self.id_names_map_ext_to_int.keys():
                self._merge_id_with_columns(id_name_ext)
            else:
                print("ID '%s' is not defined"%id)

    def _merge_id_with_columns(self, id_name_ext):
        index_levels, column_levels = self._get_index_column_levels()
        id_name_int = self.id_names_map_ext_to_int[id_name_ext]
        if id_name_int in index_levels:
            # move index 'id_name_int' to top-level in columns
            self.dataframe = self.dataframe.unstack(id_name_int).swaplevel(-2, -1, axis=1)
        else:
            print("ID '%s' is not in index"%id_name_int)
            if id_name_int in column_levels:
                print("ID '%s' is already in columns" % id_name_int)

    def unmerge_id_from_columns(self, id_names=[]):
        for id_name_ext in id_names:
            if id_name_ext in self.id_names_map_ext_to_int.keys():
                self._unmerge_id_from_columns(id_name_ext)
            else:
                print("ID '%s' is not defined"%id)

    def _unmerge_id_from_columns(self, id_name_ext):
        id_name_int = self.id_names_map_ext_to_int[id_name_ext]
        index_levels, column_levels = self._get_index_column_levels()
        if id_name_int in column_levels:
            id_level = column_levels.index(id_name_int)
            self.dataframe = self.dataframe.stack(id_level)
        else:
            print("ID name '%s' is not in index" % id_name_int)
            if id_name_int in index_levels:
                print("ID '%s' is already in index" % id_name_int)

    def flatten_column_index(self, level_separator_string=None):
        if isinstance(self.dataframe.columns, MultiIndex):
            if level_separator_string is None:
                level_separator_string = self.flattening_separator
            self.flattened_column_levels = self._get_level_names(which='columns')
            self.dataframe.columns = [level_separator_string.join(col) for col in self.dataframe.columns]
        else:
            print("Cannot flatten current columns -- only single level")

    def unflatten_column_index(self, level_names=None, level_separator_string=None):
        if not isinstance(self.dataframe.columns, MultiIndex):
            if level_names is None:
                level_names = self.flattened_column_levels
            if level_separator_string is None:
                level_separator_string = self.flattening_separator
            try:
                column_tuples = [col.split(level_separator_string) for col in self.dataframe.columns]
                self.dataframe.columns = MultiIndex.from_tuples(column_tuples, names=level_names)
                self.flattened_column_levels = []
            except Exception as e:
                print(e)
        else:
            print("Cannot expand current columns -- already has levels")

    def _get_index_column_levels(self):
        index_level_names   = self.dataframe.index.names
        column_level_names  = self.dataframe.columns.names
        return index_level_names, column_level_names

    def _get_level_names(self, which='index'):
        if which=='index':
            levels = self.dataframe.index
        elif which=='columns':
            levels = self.dataframe.columns
        return levels.names

    def _get_all_id_names_except(self, except_name='patient_id', except_name_is_external=False, ids_as_external=True):
        if except_name_is_external:
            map_dict = self.id_names_map_ext_to_int
        else:
            map_dict = self.id_names_map_int_to_ext
        map_dict.pop(except_name)
        if ids_as_external:
            id_names = list(map_dict.values())
        else:
            id_names = list(map_dict.keys())
        return id_names

    def _get_default_levels(self, mode='merge'):
        if mode=='merge':
            level_names = self._get_all_id_names_except('patient_id', except_name_is_external=False, ids_as_external=True)
        elif mode=='unmerge':
            levels = self.flattened_column_levels.copy()
            levels.pop(levels.index(self.data_name))
            level_names = [self.id_names_map_int_to_ext[level_name] for level_name in levels]
        else:
            print("Mode '%s' is not defined"%mode)
            level_names = None
        return level_names

    def get_data_as_array(self):
        return self.dataframe.values

    def get_data_as_dataframe(self):
        return self._map_dataframe_id_names(external_to_internal=False, inplace=False)

    def _update_data(self, data_array):
        data_array_orig = self.get_data_as_array()
        if data_array_orig.shape == data_array.shape:
            self.dataframe.loc[:]=data_array
        else:
            print("Dimensions of original and new data array do not agree.")


class FeatureDB(DB):
    def __init__(self, data,
                       id_int_to_ext_map = {'patient_id': 'PatientID', 'modality': 'Modality', 'roi': 'ROI'},
                       data_name = 'features',
                       flattening_separator = '_'):
        super().__init__(data, id_int_to_ext_map, data_name, flattening_separator)

    def merge_ids_into_feature_names(self, id_names=None, level_separator_string=None):
        if id_names is None:
            id_names = self._get_default_levels(mode='merge')
        elif isinstance(id_names, list):
            pass
        else:
            print("Attribute 'id_names' must be a list or 'None'")
        self.merge_ids_with_columns(id_names=id_names)
        self.flatten_column_index(level_separator_string=level_separator_string)

    def extract_ids_from_feature_names(self, id_names=None, level_separator_string=None, level_names=None):
        if id_names is None:
            id_names = self._get_default_levels(mode='unmerge')
        elif isinstance(id_names, list):
            pass
        else:
            print("Attribute 'id_names' must be a list or 'None'")
        self.unflatten_column_index(level_separator_string=level_separator_string, level_names=level_names)
        self.unmerge_id_from_columns(id_names=id_names)
        if isinstance(self.dataframe.columns, MultiIndex):
            self.flatten_column_index(level_separator_string=level_separator_string)


class OutcomeDB(DB):
    def __init__(self, data,
                       id_int_to_ext_map = {'patient_id': 'PatientID', 'modality': 'Modality', 'roi': 'ROI'},
                       data_name = 'outcomes',
                       flattening_separator = '_'):

        super().__init__(data, {}, data_name, flattening_separator)
        self._update_ids(self._get_relevant_ids(id_int_to_ext_map))

    def _get_relevant_ids(self, id_int_to_ext_map):
        columns = self.dataframe.columns.values
        id_ext_to_int_map  = {value: key for key, value in id_int_to_ext_map.items()}
        int_ids_in_columns = [id_ext_to_int_map[ext_id] for ext_id in id_ext_to_int_map.keys() if ext_id in columns]
        id_int_to_ext_map_selected = {int_id: id_int_to_ext_map[int_id] for int_id in int_ids_in_columns}
        return id_int_to_ext_map_selected
