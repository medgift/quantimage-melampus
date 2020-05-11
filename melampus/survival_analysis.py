import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter


class MelampusSurvivalAnalyzer:
    """
    MelampusSurvivalAnalyzer is used for two kinds of survival analysis:  and regression. For

    - 1. Analysis for univariate models using **Kaplan-Meier** or **Nelson-Aalen** approach
    - 2. Survival regression using **Cox's model**

        :param data: Dataset for survival analysis
        :type data: pandas.DataFrame, required
        :param time_column: The column name of the duration variable
        :type time_column: str, required
        :param event_column: The column name of the observed event variable
        :type event_column: str, required
        :param method: The name of the desired method. Options: {'kaplan_meier','nelson_aalen','cox_model'}.
        Default value: 'cox_model'

    """

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
        """
        Initializes the ``self.analyzer`` object calling the corresponding ``lifelines`` module for the desired algorithm. E.g.:
        ``method='kaplan_meier'`` the Kaplan-Meier estimator from ``lifelines`` library will be initialized.
        """
        if self.method == 'kaplan_meier':
            self.analyzer = KaplanMeierFitter()
        elif self.method == 'nelson_aalen':
            self.analyzer = NelsonAalenFitter()
        elif self.method == 'cox_model':
            self.analyzer = CoxPHFitter()
        else:
            self.analyzer = CoxPHFitter()

    def train(self):
        """
        Train the model based on desired algorithm.
        :return: The concordance index
        """
        if self.method == 'cox_model':
            try:
                self.analyzer.fit(self.data, duration_col=self.time_column, event_col=self.event_column)
            except Exception as e:
                raise Exception(str(e))
        else:
            try:
                self.analyzer.fit(self.T, self.E)
            except Exception as e:
                raise Exception(str(e))

        return self.analyzer.concordance_index_
