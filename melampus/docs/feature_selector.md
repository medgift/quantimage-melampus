
# Melampus Feature Selector
Melampus Feature Selector provides three methods explained below. The initialization is same as the preprocessor's one (see [Preprocessor](preprocessor.md) for details)

##Methods
- **variance_threshold**:   It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features.
                        :parameter (optionally) the threshold. default: 0.8
                        :returns data transformed (numpy array)

- **drop_correlated_features**: Drop all correlated features based on a specific metric and a correlation score.
                            

- **identify_correlated_features_with_target_variable**: identify features correlated with the target variable
                                                      The parameters are the same as at drop_correlated_features()
                                                      It returns a a pandas dataframe 

All transormed data are returned in numpy array format

```
from melampus.feature_selector import FeatureSelector

fs = FeatureSelector(filename= '')
x_tr = fs.variance_threshold()
fs.rfe()
x_tr = fs.drop_correlated_features(metric='pearson', score=0.9)
selected_features = fs.identify_correlated_features_with_target_variable(score=0.95, metric= 'pearson', target_var= 'volT')
```