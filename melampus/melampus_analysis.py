import pandas as pd
from melampus.database import FeatureDB, OutcomeDB
from sklearn.feature_selection import f_classif
from melampus.melampus_analysis_component import MelampusAnalysisComponent
import numpy as np

class MelampusAnalysis(object):
    def __init__(self, feature_db: FeatureDB, outcome_db: OutcomeDB, outcome_name: str,
                 missing_values_policy='drop-features'):
        """
        :param name:
        :param feature_array:
        :param outcome_array:
        :param missing_values:
            - drop-rows
            - drop-columns
            - best -> drop rows or columns so that most entries remain
            - if missing values in outcomes -> drop rows
        """
        self.feature_db = feature_db
        self.outcome_db = outcome_db
        self.set_outcome(outcome_name)
        self._init_data_arrays()
        self.analysis_components = []

    def set_outcome(self, outcome_name: str):
        if outcome_name in self.outcome_db.get_data_as_dataframe().columns:
            self.outcome_name = outcome_name
            print("Using outcome '%s'"%outcome_name)
        else:
            raise Exception("Outcome '%s' does not exist in DB"%outcome_name)

    def _init_data_arrays(self):
        self.feature_array = self.feature_db.get_data_as_array()
        self.outcome_array = self.outcome_db.get_data_as_dataframe()[self.outcome_name].values

    def add_analysis_component(self, analysis_component: MelampusAnalysisComponent):
        if analysis_component.feature_array is None:
            analysis_component.set_feature_array(self.feature_array)
        if analysis_component.outcome_array is None:
            analysis_component.set_outcome_array(self.outcome_array)
        if analysis_component.name in self._get_analysis_component_names():
            raise Exception(
                "Analysis component name '%s' already used. Names must be unique!" % analysis_component.name)
        else:
            self.analysis_components.append(analysis_component)

    def _get_analysis_component(self, name=None, index=None, include_preceding=False):
        if (name is not None) and (index is None):
            index = self._get_analysis_component_names().index(name)
        elif (index is not None) and (name is None):
            pass
        else:
            raise Exception("Only specify one of 'name', 'index'")
        if include_preceding:
            return self.analysis_components[:index + 1]
        else:
            return [self.analysis_components[index]]

    def _get_analysis_component_names(self):
        return [comp.name for comp in self.analysis_components]

    def show_analysis_components(self):
        for i, name in enumerate(self._get_analysis_component_names()):
            print("- Component %i: %s" % (i, name))

    def execute_components(self, name, rerun_preceding=True):
        if rerun_preceding:
            self._init_data_arrays()
        comps_to_run = self._get_analysis_component(name=name, include_preceding=rerun_preceding)
        for comp in comps_to_run:
            self._execute_analysis_component(comp)

    def execute_all(self):
        name_last = self._get_analysis_component_names()[-1]
        self.execute_components(name_last, rerun_preceding=True)

    def _execute_analysis_component(self, analysis_component: MelampusAnalysisComponent):
        print("Executing analysis component %s" % analysis_component.name)

