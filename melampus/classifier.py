from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate, cross_val_predict
import numpy as np

from melampus.preprocessor import Preprocessor


class MelampusClassifier(object):
    def __init__(self, filename: str, algorithm_name: str, outcomes=[], target_col=None, scaling=False, dim_red=(False, 0),
                 normalize=False):
        """
        Melampus Classifier for logistic regression. Includes all the preprocessor steps as options

        :param filename: The name of the csv file that includes the data
        :param algorithm_name: The name of the desired method. Possible values:
            - logistic_regression: For Logistic Regression
            - lasso_regression: For logistic regression with the l1 penalty
            - elastic_net: For logistic regression with the elastic net penalty
            - random_forest: For Random Forest classifier, an embedded method of decision trees
            - svm: For a Support Vector Machine classifier.
        Optional parameters:
        :param outcomes:  the outcomes as a separated dataset in list format
        :param target_col: name of the target variable if included in the csv dataset
        :param scaling: Standarization of data
        :param dim_red: For high dimensional datasets. Reduce the amount of features into a new feature space.
                        dimred[1] = number of dimentions in the new feature space
        :param normalize: Normalization with L2 data.
        """
        self.filename = filename
        self.target_col = target_col
        self.scaling = scaling
        self.dim_red = dim_red
        self.normalize = normalize
        self.data, self.outcomes = [], []
        self.outcomes = outcomes
        self.algorithm = algorithm_name
        self.classifier = object
        self.metrics = {}
        self.preprocess_data()
        self.init_classifier()

    def init_classifier(self):
        self.classifier = LogisticRegression() # default method
        if self.algorithm == 'logistic_regression':
            self.classifier = LogisticRegression()
        elif self.algorithm == 'lasso_regression':
            self.classifier = Lasso()
        elif self.algorithm == 'elastic_net':
            self.classifier = ElasticNet()
        elif self.algorithm == 'random_forest':
            self.classifier = RandomForestClassifier()
        elif self.algorithm == 'svm':
            self.classifier = SVC()

    def preprocess_data(self):
        pre = Preprocessor(filename=self.filename, target_col=self.target_col)
        if self.scaling:
            pre.standarize_data()

        if self.dim_red[0]:
            num_components = self.dim_red[1]
            pre.dimensionality_reduction(num_components=num_components)

        if self.normalize:
            pre.normalize_data()

        self.data = pre.data
        if self.target_col is not None:
            self.outcomes = pre.outcomes

    def train(self):
        # TODO: unittest train and test set exist for fitting
        print('model training ..')
        try:
            #model = cross_validate(self.classifier, self.data, self.outcomes, cv=StratifiedKFold(n_splits=5), return_estimator=True)
            self.classifier.fit(self.data, self.outcomes)
            predictions = cross_val_predict(self.classifier, self.data, self.outcomes, cv=StratifiedKFold(n_splits=5))

        except Exception as e:
            raise Exception('classifier_train - Exception error: {}'.format(str(e)))
        self.calculate_assessment_metrics(predictions)

    def calculate_assessment_metrics(self, predictions= int):

        pass