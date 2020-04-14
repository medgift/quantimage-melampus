# Melampus Preprocessor
Melampus Preprocessor accepts a csv file with the **column names must be included**. Also, the dataset must contains a separate column named exactly **'PatientID'** with the samples ids. 
The name of the target variable can also be given as an optional parameter in case that the target variable is included in the csv file.

##Methods
After the initialization, three function are provided for standarization, normalization and dimensionality reduction of the data:
- **standarize_data**: standarization of data
- **normalize_data**: normalization of data with L2 norm
- **dimensionality_reduction**: For high dimensional datasets. It reduces the amount of features into a new feature space. 
The dimensions of the new space must be declared with the parameter 'num_components'. 

You can access the transformed data on **pre.data** (format: numpy array)

```
from melampus.preprocessor import Preprocessor

pre = Preprocessor(filename='../synthetic_data/output_L0_GTVL.csv')
pre.standarize_data()
pre.normalize_data()
pre.dimensionality_reduction(num_components=5)
print('Processed data: {}'.format(pre.data))
```

##Initialization parameters
The user must first initialize her classifier and then to use the train function to fit the data into the selected model.
###Required parameters
+ **filename** : The name of the csv file that includes the data

###Optional parameters
+ **target_col**: name of the target variable if included in the csv dataset
       
