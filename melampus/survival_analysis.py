import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter


class MelampusSurvivalAnalysis:
    def __init__(self, data: pd.DataFrame, time_column: str, event_column: str, method='kaplan_meier'):
        self.data = data
        self.time_column = time_column
        self.event_column = event_column
        self.method = method
        self.T = pd.Series  # time variable
        self.E = pd.Series  # event variable
        self.classifier = object
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
            self.classifier = KaplanMeierFitter()
        elif self.method == 'nelson_aalen':
            self.classifier = NelsonAalenFitter()
        else:
            self.classifier = KaplanMeierFitter()

    def train(self):
        try:
            self.classifier.fit(self.T, self.E)
        except Exception as e:
            raise Exception(str(e))

    def assess(self):
        pass
