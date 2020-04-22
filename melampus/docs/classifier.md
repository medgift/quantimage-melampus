# Melampus classifier
Melampus classifier gives users the option to train a supervised model on image datasets excracted by Kheops platform. It  It provides as results metrics about its efficacy and accuracy. Especially, f1 score and confusion matrix are 
provided to the user into the classifier object.

The initialization of a model contains two required input parameters. 
It also includes all the preprocessor steps as optional parameters (see [Preprocessor](preprocessor.md)).

e.g.: 
```
from melampus.classifier import MelampusClassifier

mel_clf = MelampusClassifier(filename='synthetic_data/all.csv', algorithm_name='elastic_net',
                             target_col='label', normalize=True, scaling=True, dim_red=(True, 5))
mel_clf.train()
accuracy =mel_clf.metrics['accuracy']
precision = mel_clf.metrics['precision']
area_under_curve = mel_clf.metrics['area_under_curve']
recall = mel_clf.metrics['recall']
true_positives = mel_clf.metrics['true_pos']
true_negatives = mel_clf.metrics['true_neg']
false_positives = mel_clf.metrics['false_pos']
false_negatives = mel_clf.metrics['false_neg']
```



##Initialization parameters
The user must first initialize her classifier and then to use the train function to fit the data into the selected model.
###Required parameters
+ **filename** : The name of the csv file that includes the data
+ **algorithm_name** : The name of the desired method. Possible values:
    - "logistic_regression": For Logistic Regression
    - "lasso_regression": For logistic regression with the l1 penalty
    - "elastic_net": For logistic regression with the elastic net penalty
    - "random_forest": For Random Forest classifier, an embedded method of decision trees
    - "svm": For a Support Vector Machine classifier.

###Optional parameters
+ **outcomes**:  the outcomes as a separated dataset in list format
+ **target_col**: name of the target variable if included in the csv dataset
+ **scaling**: Standarization of data
+ **dim_red**: For high dimensional datasets. Reduce the amount of features into a new feature space.
                dimred[1] = number of dimentions in the new feature space
+ **normalize**: Normalization with L2 data.
        
