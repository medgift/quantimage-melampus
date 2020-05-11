import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter


class MelampusSurvivalAnalyzer:
    def __init__(self, data: pd.DataFrame, time_column: str, event_column: str, method='cox_model'):
        self.data = data
        self.time_column = time_column
        self.event_column = event_column
        self.method = method
        self.T = pd.Series  # time variable
        self.E = pd.Series  # event variable
        self.analyzer = object
        if self.method != 'cox_model':
            self.init_columns()
        self.init_survival_classifier()

    def init_columns(self):
        try:
            self.T = self.data[self.time_column]
            self.E = self.data[self.event_column]
        except:
            raise KeyError('column name is not valid')

    def init_survival_classifier(self):
        if self.method == 'kaplan_meier':
            self.analyzer = KaplanMeierFitter()
        elif self.method == 'nelson_aalen':
            self.analyzer = NelsonAalenFitter()
        elif self.method == 'cox_model':
            self.analyzer = CoxPHFitter()
        else:
            self.analyzer = CoxPHFitter()

    def train(self):
        if self.method == 'cox_model':
            try:
                self.analyzer.fit(self.data, duration_col = self.time_column, event_col= self.event_column)
            except Exception as e:
                raise Exception(str(e))
        else:
            try:
                self.analyzer.fit(self.T, self.E)
            except Exception as e:
                raise Exception(str(e))

    def assess(self):
        return self.analyzer.concordance_index_
