from time import time

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_predict,
    train_test_split,
    RepeatedKFold,
)
from sklearn.svm import SVC
from melampus.preprocessor import MelampusPreprocessor


class MelampusClassifier:
    """
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
    """

    def __init__(
        self,
        filename: str,
        algorithm_name: str,
        outcomes=[],
        target_col=None,
        scaling=False,
        dim_red=(False, 0),
        normalize=False,
    ):
        self.filename = filename
        self.target_col = target_col
        self.scaling = scaling
        self.dim_red = dim_red
        self.normalize = normalize
        self.data, self.outcomes = [], []
        self.num_cases = int
        self.num_cases_in_each_class = []
        self.outcomes = outcomes
        self.algorithm = algorithm_name
        self.classifier = object
        self.metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "area_under_curve": None,
            "true_pos": None,
            "false_pos": None,
            "true_neg": None,
            "false_neg": None,
        }
        self.preprocess_data()
        self.init_classifier()
        self.regression_methods = ["lasso_regression", "elastic_net"]

    def init_classifier(self):
        """
        Initializes the ``self.classifier`` object calling the corresponding sklearn module for the desired algorithm. E.g.:
        ``algorithm='random_forest'`` a Random Forest classifier from ``scikit-learn`` library will be trained.
        """

        self.classifier = LogisticRegression()  # default method
        if self.algorithm == "logistic_regression":
            self.classifier = LogisticRegression()
        elif self.algorithm == "lasso_regression":
            self.classifier = LogisticRegression(penalty="l1", solver="saga")
        elif self.algorithm == "elastic_net":
            self.classifier = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5
            )
        elif self.algorithm == "random_forest":
            self.classifier = RandomForestClassifier()
        elif self.algorithm == "svm":
            self.classifier = SVC(probability=True)

    def preprocess_data(self):
        """
        Preprocessing of the data using :class:`melampus.preprocessor.MelampusPreprocessor`.
        """

        pre = MelampusPreprocessor(
            filename=self.filename, target_col=self.target_col, outcomes=self.outcomes
        )
        if self.scaling:
            pre.standarize_data()

        if self.dim_red[0]:
            num_components = self.dim_red[1]
            pre.dimensionality_reduction(num_components=num_components)

        if self.normalize:
            pre.normalize_data()

        self.data = pre.data
        self.num_cases = pre.num_cases
        self.num_cases_in_each_class = pre.num_cases_in_each_class
        if self.target_col is not None:
            self.outcomes = pre.outcomes

    def train(self):
        """
        Simple training of a classifier model to fit on **the whole** dataset; without cross-validation or evaluations.

        :return: The trained model (type: object)
        """

        try:
            print("classifier training (method: {})..".format(self.algorithm))
            t0 = time()
            self.classifier.fit(self.data, self.outcomes)
        except Exception as e:
            print(str(e))
            raise

        print(
            "classifier was trained without cv or evaluation in {} sec".format(
                time() - t0
            )
        )
        return self.classifier

    def train_with_cv(self, test_size: float):
        """
        Training of a classifier model with holding out a specific part of the available data as a test set.
        For instance to train a model with 70% of the dataset, and then test it out with the remaining 30%.

        :param test_size: proportion of the samples the user wants to leave out. It must be provided as a float number.
        :type test_size: float, required

        So, for example, if we split 70%-30%, the user must provide the amount of 30% as a float number: 0.3

        :return:
            - The trained model (type: object)
            - The test set: the samples that were holded out as a test set
        """

        x_train, x_test, y_train, y_test = train_test_split(
            self.data, self.outcomes, test_size=test_size
        )
        try:
            print("classifier training (method: {})..".format(self.algorithm))
            t0 = time()
            self.classifier.fit(x_train, y_train)
        except Exception as e:
            print(str(e))
            raise

        print(
            "classifier was trained with CV (train: {0}% - test: {1}%)) in {2} sec".format(
                (1 - test_size) * 100, test_size * 100, time() - t0
            )
        )
        return self.classifier, x_test

    def train_and_evaluate(self, leave_one_out=False, random_state=None):
        """
        Training of the initialized model with repeated k-fold cross-validation. Then, we calculate some assessment metrics for the
        trained model using :meth:`melampus.classifier.MelampusClassifier.calculate_assessment_metrics` method.

        :param leave_one_out: Leave one out method for cross-validation. default value=False
        :type leave_one_out: bool, optional
        :param random_state: Random state for allowing reproducibility, default value=None
        :type random_state: int, optional
        :return: The trained model (type: object)

        :raise Exception: If the number of samples in the training dataset is less than 5.

        *Note about the cross-validation*: The selection of k is also done dynamically. If a class has less than 5 cases,
        then k takes the value of that number. E.g.: in a dataset of 10 cases, with 8 cases were assigned to 1 and 2 cases to 0, then k=2
        """

        print("classifier training (method: {})..".format(self.algorithm))
        t0 = time()
        k = 5
        repetitions = 10
        validator = KFold(n_splits=k, random_state=random_state)

        minimum_number_of_cases = min(
            [i for i in self.num_cases_in_each_class.values()]
        )
        if minimum_number_of_cases < 5:
            k = minimum_number_of_cases

        if self.num_cases > 5:
            if leave_one_out:
                k = self.num_cases

            predictions = cross_val_predict(
                self.classifier, self.data, self.outcomes, cv=validator
            )
        else:
            raise Exception(
                "No sense to train a predictive model, number of cases are less than 5.."
            )
        self.calculate_assessment_metrics(predictions)
        print(
            "classifier was trained with {0}-fold CV and evaluations in {1} sec".format(
                k, (time() - t0)
            )
        )
        return self.classifier, f"kfold", {"k": k}

    def predict(self, samples: list, predict_probabilities=False):
        """
        Method to use a trained model to make predictions on new samples.

        :param samples: The new samples on which we want to make predictions
        :type samples: list, required
        :param predict_probabilities: If True, the method returns probability predictions **for each class seperately** instead of binary ones
        :type predict_probabilities: bool, optional - default value = False
        :return: The predictions (or probabilities). Type: **list** (or **list of lists** for probabilities of each class)
        """

        if predict_probabilities:
            return self.classifier.predict_proba(samples)
            # return self.classifier.predict_log_proba_(samples)

        return self.classifier.predict(samples)

    def calculate_assessment_metrics(self, predictions: list):
        """
        Calculation of assessment metrics using the corresponding scikit-learn modules. The predictions on which the model
        is being assessed are calculated on the test samples derived by each of the cross-validation iterations.

        :param predictions: A list with classifier predictions on the test samples
        :type predictions: list, required
        The results are stored in self.metrics object (dictionary).
        """
        self.metrics["accuracy"] = metrics.accuracy_score(self.outcomes, predictions)
        self.metrics["precision"] = metrics.precision_score(self.outcomes, predictions,)
        self.metrics["recall"] = metrics.recall_score(self.outcomes, predictions)
        self.metrics["area_under_curve"] = metrics.roc_auc_score(
            self.outcomes, predictions
        )
        (
            self.metrics["true_neg"],
            self.metrics["false_pos"],
            self.metrics["false_neg"],
            self.metrics["true_pos"],
        ) = metrics.confusion_matrix(self.outcomes, predictions).ravel()
