from time import time

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from melampus.preprocessor import MelampusPreprocessor


class MelampusClassifier:
    '''
    This is a machine learning classifier for datasets from medical images.
    The initialization of the melampus classifier object contains two required input parameters and some optional
    parameters inherited from :class:`melampus.preprocessor.MelampusPreprocessor`.

    :param filename: The name of the csv file that includes the data
    :type filename: str, required
    :param algorithm_name: The name of the desired method. Possible values:
        - logistic_regression: For Logistic Regression
        - lasso_regression: For logistic regression with the l1 penalty
        - elastic_net: For logistic regression with the elastic net penalty
        - random_forest: For Random Forest classifier, an embedded method of decision trees
        - svm: For a Support Vector Machine classifier.
    :type algorithm_name: str, required
    :param outcomes:  the outcomes as a separated entry (e.g.: [0,1,1,1,1,0...,1]), defaults to []
    :type outcomes:  list, optional
    :param target_col: name of the target variable if included in the csv dataset, defaults to None
    :type target_col: str, opional
    :param scaling: Standarization of data, defaults to False
    :type scaling: bool, optional
    :param dim_red: For high dimensional datasets. Reduce the amount of features into a new feature space.
                    dimred[1] = number of dimentions in the new feature space. defaults to (False, 0)
    :type dim_red: tuple, optional
    :param normalize: Normalization of data with L2 norm.
    :type normalize: bool, optional, defaults to False
    '''
    def __init__(self, filename: str, algorithm_name: str, outcomes=[], target_col=None, scaling=False,
                 dim_red=(False, 0),
                 normalize=False):
        self.filename = filename
        self.target_col = target_col
        self.scaling = scaling
        self.dim_red = dim_red
        self.normalize = normalize
        self.data, self.outcomes = [], []
        self.outcomes = outcomes
        self.algorithm = algorithm_name
        self.classifier = object
        self.metrics = {'accuracy': None, 'precision': None, 'recall': None, 'area_under_curve': None, 'true_pos': None,
                        'false_pos': None, 'true_neg': None, 'false_neg': None}
        self.preprocess_data()
        self.init_classifier()
        self.regression_methods = ['lasso_regression', 'elastic_net']

    def init_classifier(self):
        '''
        Initializes the classifier object calling the corresponding sklearn module for the desired algorithm. E.g.:
        ``algorithm='random_forest'`` a Random Forest classifier from scikit-learn library will be trained.
        '''
        self.classifier = LogisticRegression()  # default method
        if self.algorithm == 'logistic_regression':
            self.classifier = LogisticRegression()
        elif self.algorithm == 'lasso_regression':
            self.classifier = LogisticRegression(penalty='l1', solver='saga')
        elif self.algorithm == 'elastic_net':
            self.classifier = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
        elif self.algorithm == 'random_forest':
            self.classifier = RandomForestClassifier()
        elif self.algorithm == 'svm':
            self.classifier = SVC()

    def preprocess_data(self):
        '''
        Preprocessing of the data using :class:`melampus.preprocessor.MelampusPreprocessor`.
        '''
        pre = MelampusPreprocessor(filename=self.filename, target_col=self.target_col)
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
        '''
        Training of the initialized model with cross-validation. Then, we calculate some assessment metrics for the
        trained model using :meth:`melampus.classifier.MelampusClassifier.calculate_assessment_metrics` method.
        For the model's training, we use the StratifiedKFold cv technique for imbalanced data.
        '''
        print('classifier training (method: {})..'.format(self.algorithm))
        t0 = time()
        predictions = cross_val_predict(self.classifier, self.data, self.outcomes, cv=StratifiedKFold(n_splits=5))
        print('classifier was trained in {} sec'.format(time() - t0))
        self.calculate_assessment_metrics(predictions)

    def calculate_assessment_metrics(self, predictions: list):
        '''
        Calculation of assessment metrics using the corresponding scikit-learn modules. The predictions on which the model
        is being assessed are calculated on the test samples derived by each of the cross-validation iterations.

        :param predictions: A list with classifier predictions on the test samples
        :type predictions: list, required
        The results are stored in self.metrics object (dictionary).
        '''
        self.metrics['area_under_curve'] = metrics.roc_auc_score(self.outcomes, predictions)
        self.metrics['accuracy'] = metrics.accuracy_score(self.outcomes, predictions)
        self.metrics['precision'] = metrics.precision_score(self.outcomes, predictions)
        self.metrics['recall'] = metrics.recall_score(self.outcomes, predictions)
        self.metrics['true_neg'], self.metrics['false_pos'], self.metrics['false_neg'], self.metrics[
            'true_pos'] = metrics.confusion_matrix(self.outcomes, predictions).ravel()
