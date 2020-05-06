Examples
=============
Preprocessor
*************
.. code-block:: python

    """
    Melampus Preprocessor accepts a csv file with column names. And (optionally) the name of target variable included in the csv file.
    Functions:
    - standarize_data: standarization of data
    - normalize_data: normalization of data with L2 norm
    - dimensionality_reduction: For high dimensional datasets. Reduce the amount of features into a new feature space.
            :parameter number of dimentions in the new feature space

    You can access the transformed dataset on pre.data (format: numpy array)
    """

    from melampus.preprocessor import MelampusPreprocessor

    pre = MelampusPreprocessor(filename='../synthetic_data/output_L0_GTVL.csv', )
    pre.standarize_data()
    pre.normalize_data()
    pre.dimensionality_reduction(num_components=5)
    print('Processed data: {}'.format(pre.data))

Classifier
*******************
.. code-block:: python
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

    Classifier provides as results the coefficients and the intercepts of the Logistic Regression. These parameters
    are included in mel_clf object (e.g.: mel_clf.coefficients)
    """

    from melampus.classifier import MelampusClassifier

    outcomes = [0, 1] * 6
    mel_clf = MelampusClassifier(filename='synthetic_data/all.csv', algorithm_name='logistic_regression', target_col='label',
                                 normalize=True, scaling=True, dim_red=(True, 5))
    mel_clf.train()
    accuracy =mel_clf.metrics['accuracy']
    precision = mel_clf.metrics['precision']


Feature Selector
*******************
.. code-block:: python
    """
    Melampus Feature Selector accepts a csv file with column names. And (optionally) the name of target variable included in the csv file.
    Functions:
    - variance_threshold:   It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features.
                            :parameter (optionally) the threshold. default: 0.8
                            :returns data transformed (numpy array)
    - drop_correlated_features: Drop all correlated features based on a specific metric and a correlation score
                                :returns pandas dataframe
    - identify_correlated_features_with_target_variable:  identify features correlated with the target variable
                                                          :parameters same as drop_correlated_features()
                                                          :returns a pandas dataframe

    """

    from melampus.feature_selector import MelampusFeatureSelector

    fs = MelampusFeatureSelector(filename='synthetic_data/output_L0_GTVL.csv')
    x_tr = fs.variance_threshold()
    fs.rfe()
    x_tr = fs.drop_correlated_features(metric='pearson', score=0.9)
    selected_features = fs.identify_correlated_features_with_target_variable(score=0.95, metric= 'pearson', target_var= 'volT')
