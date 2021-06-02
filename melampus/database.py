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
        self.df_index_status = 'original'

        # initialize attributes used later
        self.flattened_column_levels = []
        self.flattening_separator = flattening_separator
        self.data_name = data_name
        self._update_ids(id_int_to_ext_map=id_int_to_ext_map)

    def _update_ids(self, id_int_to_ext_map):
        # set ids
        self.id_names_map_int_to_ext = id_int_to_ext_map
        self.id_names_map_ext_to_int = {value: key for key, value in self.id_names_map_int_to_ext.items()}

        # rename ID columns & set index
        self.dataframe.columns.names = [self.data_name]
        # map ids from original ids in csv file to standardized identifiers
        self._map_dataframe_index_to_internal(inplace=True)

    def _get_mapped_ids(self, ids, external_to_internal=True):
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(ids, list):
            ids = [id for id in ids if not id is None]
            if len(ids)>0:
                if external_to_internal:
                    map = self.id_names_map_ext_to_int
                else:
                    map = self.id_names_map_int_to_ext
                ids_in_keys = [id in map.keys() for id in ids]
                if all(ids_in_keys):
                    mapped_ids = [map[id] for id in ids]
                    if len(mapped_ids)==1:
                        return mapped_ids[0]
                    else:
                        return mapped_ids
                else:
                    raise Exception("Cannot map all ids")
            else:
                return []
        else:
            raise Exception("Expect argument 'ids' to be of type string or list of strings.")

    def _get_internal_ids(self, ids):
        return self._get_mapped_ids(ids, external_to_internal=True)

    def _get_external_ids(self, ids):
        return self._get_mapped_ids(ids, external_to_internal=False)

    def _get_level_names(self, which='index'):
        if which == 'index':
            levels = self.dataframe.index
        elif which == 'columns':
            levels = self.dataframe.columns
        return levels.names

    def _get_available_indices(self, id_as_external=True):
        columns = list(self.dataframe.reset_index().drop(columns=['index']).columns)
        if self.df_index_status=='original':
            indices_avail = [idx_var for idx_var in self.id_names_map_ext_to_int.keys() if idx_var in columns]
            indices_avail = self._get_internal_ids(indices_avail)
        elif self.df_index_status=='internal':
            indices_avail = [idx_var for idx_var in self.id_names_map_int_to_ext.keys() if idx_var in columns]
        if id_as_external:
            indices_avail = self._get_external_ids(indices_avail)
        return indices_avail

    def _get_current_indices(self, ids_as_external=True):
        levels = [level for level in self._get_level_names(which='index') if not level is None]
        if len(levels)==0:
            if self.df_index_status=='internal':
                levels = self._get_available_indices(id_as_external=False)
            elif self.df_index_status=='original':
                levels = self._get_available_indices(id_as_external=True)
        if ids_as_external and self.df_index_status=='internal':
            levels = self._get_external_ids(levels)
        elif (not ids_as_external) and self.df_index_status=='original':
            levels = self._get_internal_ids(levels)
        return levels

    def _map_dataframe_id_names(self, external_to_internal=True, inplace=True):
        if (self.df_index_status=='internal') and external_to_internal:
            print("Dataframe already uses internal indices")
        elif (self.df_index_status=='original') and not external_to_internal:
            print("Dataframe already uses original indices")
        else:
            if external_to_internal:
                relevant_indices = self._get_current_indices(ids_as_external=True)
                map = { key:value for key, value in self.id_names_map_ext_to_int.items() if key in relevant_indices}
            else:
                relevant_indices = self._get_current_indices(ids_as_external=False)
                map = { key:value for key, value in self.id_names_map_int_to_ext.items() if key in relevant_indices}
            df = self.dataframe.reset_index(drop=False, inplace=False)
            if 'index' in df.columns:
                df.drop(columns=['index'], inplace=True)
            df.rename(columns=map, inplace=True)
            if len(map) > 0:
                df.set_index(list(map.values()), inplace=True)
            if inplace:
                self.dataframe = df
            else:
                return df

    def _map_dataframe_index_to_original(self, inplace=True):
        df = self._map_dataframe_id_names(external_to_internal=False, inplace=inplace)
        if inplace:
            self.df_index_status = 'original'
        return df

    def _map_dataframe_index_to_internal(self, inplace=True):
        df = self._map_dataframe_id_names(external_to_internal=True, inplace=inplace)
        if inplace:
            self.df_index_status = 'internal'
        return df

    def _get_flattened_level_names(self, ids_as_external=True):
        levels = self.flattened_column_levels.copy()
        if len(levels)>0:
            if ids_as_external and (self.df_index_status=='internal'):
                level_names = self._get_external_ids(levels)
            elif (not ids_as_external) and (self.df_index_status=='original'):
                level_names = self._get_internal_ids(levels)
        else:
            level_names = []
        return level_names

    def _get_unflattened_level_names(self, ids_as_external=True):
        all_levels = self.id_names_map_int_to_ext.keys()
        flattened_levels = self._get_flattened_level_names(ids_as_external=False)
        unflattened_levels = [level for level in all_levels if not level in  flattened_levels]
        if ids_as_external:
            unflattened_levels = self._get_external_ids(unflattened_levels)
        return unflattened_levels

    def merge_ids_with_columns(self, id_names=[]):
        for id_name_ext in id_names:
            if id_name_ext in self.id_names_map_ext_to_int.keys():
                self._merge_id_with_columns(id_name_ext)
            else:
                raise Exception("ID '%s' is not defined"%id_name_ext)

    def _merge_id_with_columns(self, id_name_ext):
        id_name_int = self._get_internal_ids(id_name_ext)
        if id_name_int in self._get_level_names(which='index'):
            # move index 'id_name_int' to top-level in columns
            self.dataframe = self.dataframe.unstack(id_name_int).swaplevel(-2, -1, axis=1)
        else:
            if id_name_int in self._get_level_names(which='columns'):
                print("ID '%s' is already in columns" % self._get_external_ids(id_name_int))
            raise Exception("ID '%s' is not in index"% self._get_external_ids(id_name_int))

    def unmerge_ids_from_columns(self, id_names=[]):
        for id_name_ext in id_names:
            if id_name_ext in self.id_names_map_ext_to_int.keys():
                self._unmerge_id_from_columns(id_name_ext)
            else:
                raise Exception("ID '%s' is not defined"%id_name_ext)

    def _unmerge_id_from_columns(self, id_name_ext):
        id_name_int = self._get_internal_ids(id_name_ext)
        column_levels = self._get_level_names(which='columns')
        if id_name_int in column_levels:
            id_level = column_levels.index(id_name_int)
            self.dataframe = self.dataframe.stack(id_level)
        else:
            if id_name_int in self._get_level_names(which='index'):
                print("ID '%s' is already in index" % id_name_int)
            raise Exception("ID '%s' is not merged in columns" % self._get_external_ids(id_name_int))

    def flatten_column_index(self, level_separator_string=None):
        if isinstance(self.dataframe.columns, MultiIndex):
            if level_separator_string is None:
                level_separator_string = self.flattening_separator
            levels = [level_name for level_name in self._get_level_names(which='columns') if not level_name is None]
            if self.data_name in levels:
                levels.pop(levels.index(self.data_name))
            self.flattened_column_levels = self.flattened_column_levels + levels
            self.dataframe.columns = [level_separator_string.join(col) for col in self.dataframe.columns]
        else:
            raise Exception("Cannot flatten current columns -- only single level")

    def unflatten_column_index(self, level_names=None, level_separator_string=None):
        if not isinstance(self.dataframe.columns, MultiIndex):
            if level_names is None:
                level_names = self.flattened_column_levels
            if level_separator_string is None:
                level_separator_string = self.flattening_separator
            column_tuples = [col.split(level_separator_string) for col in self.dataframe.columns]
            self.dataframe.columns = MultiIndex.from_tuples(column_tuples, names=level_names+[self.data_name])
            self.flattened_column_levels = [flattened_level for flattened_level in self.flattened_column_levels if not flattened_level in level_names]
        else:
            raise Exception("Cannot expand current columns -- already has levels")

    def _get_current_id_names_except(self, exclude_names=['patient_id'], except_name_is_external=False, ids_as_external=True):
        current_id_names = self._get_current_indices(ids_as_external=except_name_is_external)
        selected_id_names = [id_name for id_name in current_id_names if not id_name in exclude_names]
        if ids_as_external:
            selected_id_names = self._get_external_ids(selected_id_names)
        return selected_id_names

    def _get_default_levels(self, mode='merge'):
        if mode=='merge':
            level_names = self._get_current_id_names_except(exclude_names=['patient_id'],
                                                            except_name_is_external=False, ids_as_external=True)
        elif mode=='unmerge':
            level_names = self._get_flattened_level_names(ids_as_external=True)
        else:
            raise Exception("Mode '%s' is not defined"%mode)
            level_names = None
        return level_names

    def get_data_as_array(self):
        return self.dataframe.values

    def get_data_as_dataframe(self):
        return self._map_dataframe_index_to_original(inplace=False)

    def _update_data(self, data_array):
        data_array_orig = self.get_data_as_array()
        if data_array_orig.shape == data_array.shape:
            self.dataframe.loc[:]=data_array
        else:
            raise Exception("Dimensions of original and new data array do not agree.")

    def export_dataframe(self, path_to_file, orig_ids=True):
        if orig_ids:
            self._map_dataframe_index_to_original(inplace=True)
        super().export_dataframe(path_to_file)
        if orig_ids:
            self._map_dataframe_index_to_internal(inplace=True)


class FeatureDB(DB):
    def __init__(self, data,
                       id_int_to_ext_map = {'patient_id': 'PatientID', 'modality': 'Modality', 'roi': 'ROI'},
                       data_name = 'features',
                       flattening_separator = '_'):
        super().__init__(data, id_int_to_ext_map, data_name, flattening_separator)

    def get_feature_names(self):
        return self.get_data_as_dataframe().columns

    def get_n_features(self):
        return len(self.get_feature_names())

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
        print('flattened column levels before: ', self.flattened_column_levels)
        if id_names is None:
            id_names = self._get_default_levels(mode='unmerge')
        elif isinstance(id_names, list):
            pass
        else:
            print("Attribute 'id_names' must be a list or 'None'")
        self.unflatten_column_index(level_separator_string=level_separator_string, level_names=level_names)
        print('flattened column levels after: ', self.flattened_column_levels)
        self.unmerge_ids_from_columns(id_names=id_names)
        if isinstance(self.dataframe.columns, MultiIndex):
            self.flatten_column_index(level_separator_string=level_separator_string)


class OutcomeDB(DB):
    def __init__(self, data,
                       id_int_to_ext_map = {'patient_id': 'PatientID', 'modality': 'Modality', 'roi': 'ROI'},
                       data_name = 'outcomes',
                       flattening_separator = '_'):

        super().__init__(data, id_int_to_ext_map, data_name, flattening_separator)

    def _assert_categorical(self, outcome_name, order=None):
        if outcome_name in self.dataframe.columns:
            s = self.dataframe[outcome_name].astype('category')
            if order is not None:
                s.cat.set_categories(order, ordered=True)
            self.dataframe[outcome_name] = s

    def _expand_categorical(self, outcome_name):
        if outcome_name in self.dataframe.columns:
            if self.dataframe[outcome_name].dtype.name != 'category':
                print("Outcome '%s' not categorical, trying to convert"%outcome_name)
                self._assert_categorical(outcome_name=outcome_name)
            outcome_name_codes = "%s_codes"%outcome_name
            self.dataframe[outcome_name_codes] = self.dataframe[outcome_name].cat.codes