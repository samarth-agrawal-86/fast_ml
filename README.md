## Fast-ML is a Python package with numerous inbuilt functionalities to make the life of a data scientist much easier
fast_ml follow Scikit-learn type functionality with fit() and transform() methods to first learn the transforming parameters from training dataset and then transforms the training/validation/test dataset

**Important Note :** You learn the parameter by applying fit() method ONLY on train method and then apply transform on train/valid/test dataset. Be it Missing Value Imputation, Outliers, Feature Engineering for Numerical/Categorical ... Parameters are learned from the training dataset on which the model trains. 


### Installing
```python
pip install fast_ml
```

## Table of Contents:
1. **Utilities**
2. **Exploratory Data Analysis (EDA)**
3. **Missing Data Analysis**
4. **Missing Data Imputation**
5. **Outlier Treatment**
6. **Feature Engineering**
7. **Model Evaluation**
8. **Feature Selection**


## Glossary
> * df : Dataframe, refers to dataset used for analysis
> * variable : str, refers to a single variable. As required in the function it has to be passed ex 'V1'
> * variables : list type, refers to list of variables. Must be passed as list ex ['V1', 'V2]. Even a single variable has to be passed in list format. ex ['V1']
> * target : str, refers to target variable 
> * model : str, ML problem type. use 'classification' or 'clf' for classification problems and 'regression' or 'reg' for regression problems
> * method : str, refers to various techniques available for Missing Value Imputation, Feature Engieering... as available in each module

## 1. Utilities

```python
from fast_ml.utilities import reduce_memory_usage, display_all

# reduces the memory usage of the dataset by optimizing for the datatype used for storing the data
train = reduce_memory_usage(train, convert_to_category=False)
```
1. **reduce_memory_usage**(*df, convert_to_category = False*)
    * This function reduces the memory used by dataframe
2. **display_all**(*df*)
    * Use this function to show all rows and all columns of dataframe. By default pandas only show top and bottom 20 rows, columns

## 2. Exploratory Data Analysis (EDA)

```python
from fast_ml import eda
```

### 2.1) Overview
```python
from fast_ml import eda

train = pd.read_csv('train.csv')

# One of the most useful dataframe summary view
summary_df = eda.df_summary(train)
display_all(summary_df)
```
1. **eda.df_info**(*df*)
    * Returns a dataframe with useful summary - variables, datatype, number of unique values, sample of unique values, missing count, missing percent
2. **eda.df_cardinality_info**(*df, raw_data = True*)
    * Returns a dataframe with useful summary - variables, datatype, number of unique values, sample of unique values, missing count, missing percent
3. **eda.df_missing_info**(*df, raw_data = True*)
    * Returns a dataframe with useful summary - variables, datatype, number of unique values, sample of unique values, missing count, missing percent   

### 2.2) Numerical Variables
```python
from fast_ml import eda

train = pd.read_csv('train.csv')

#one line of command to get commonly used plots for all the variables provided to the function
eda.numerical_plots_with_target(train, num_vars, target, model ='clf')
```
1. **eda.numerical_describe**(*df, variables=None, method='10p'*)
    * Dataframe with variouls count, mean, std and spread statistics for all the variables passed in input
2. **eda.numerical_variable_detail**(*df, variable, model = None, target=None, threshold = 20*)
    * Various summary statistics, spread statistics, outlier, missing values, transformation diagnostic... a detailed analysis for a single variable provided as input
3. **eda.numerical_plots**(*df, variables, normality_check = False*)
    * Uni-variate plots - Variable Distribution of all the numerical variables provided as input with target. Can also get the Q-Q plot for assessing the normality
4. **eda.numerical_plots_with_target**(*df, variables, target, model*)
    * Bi-variate plots - Scatter plot of all the numerical variables provided as input with target.
5. **eda.numerical_check_outliers**(*df, variables=None, tol=1.5, print_vars = False*)
6. **eda.numerical_bins_with_target**(*df, variables, target, model='clf', create_buckets = True, method='5p', custom_buckets=None*)
    * Useful for deciding the suitable binning for numerical variable. Displays 2 graphs 'overall event rate' & 'within category event rate'

### 2.3) Categorical Variables
```python
from fast_ml import eda

train = pd.read_csv('train.csv')

#one line of command to get commonly used plots for all the variables provided to the function
eda.categorical_plots_with_target(train, cat_vars, target, add_missing=True, rare_tol=5)
```
1. **eda.categorical_variable_detail**(*df, variable, model = None, target=None,  rare_tol=5*)
    * Various summary statistics, missing values, distributions ... a detailed analysis for a single variable provided as input
2. **eda.categorical_plots**(*df, variables, add_missing = True, add_rare = False, rare_tol=5*)
    * Uni-variate plots - distribution of all the categorical provided as input
3. **eda.categorical_plots_with_target**(*df, variables, target, model='clf', add_missing = True,  rare_tol1 = 5, rare_tol2 = 10*)
    * Bi-variate plots - distribution of all the categorical provided as input with target
4. **eda.categorical_plots_with_rare_and_target**(*df, variables, target, model='clf', add_missing=True, rare_tol1=5, rare_tol2=10*)
    * Bi-variate plots - distribution of all the categorical provided as input with target with 2 inputs as rare threshold. Useful for deciding the rare bucketing
5. **eda.categorical_plots_for_miss_and_freq**(*df, variables, target, model = 'reg'*)
    * Uni-variate plots - distribution of all the categorical provided as input with target with 2 inputs as rare threshold. Useful for deciding the rare bucketing

## 3. Missing Data Analysis

```python
from fast_ml.missing_data_analysis import MissingDataAnalysis
```
### 2.1) Class MissingDataAnalysis 
1. explore_numerical_imputation (variable)
2. explore_categorical_imputation (variable)


## 4. Missing Data Imputation

```python
from fast_ml.missing_data_imputation import MissingDataImputer_Numerical, MissingDataImputer_Categorical
```
### 4.1) class MissingDataImputer_Numerical 

```python
from fast_ml.missing_data_imputation import MissingDataImputer_Numerical

train = pd.read_csv('train.csv')

num_imputer = MissingDataImputer_Numerical(df, method = 'median')

#Scikit-learn type fit() transform() functionality
# Use fit() only on the train dataset
num_imputer.fit(train, num_vars)

# Use transform() on train/test dataset
train = num_imputer.transform(train)
test = num_imputer.transform(test)
```
* Methods:
  - 'mean'
  - 'median'
  - 'mode'
  - 'custom_value'
  - 'random'
1. **fit**(*df, num_vars*)
2. **transform**(*df*)

### 4.2) class MissingDataImputer_Categorical
```python
from fast_ml.missing_data_imputation import MissingDataImputer_Categorical

train = pd.read_csv('train.csv')

cat_imputer = MissingDataImputer_Categorical(df, method = 'frequent')

#Scikit-learn type fit() transform() functionality
# Use fit() only on the train dataset
cat_imputer.fit(train, cat_vars)

# Use transform() on train/test dataset
train = cat_imputer.transform(train)
test = cat_imputer.transform(test)
```
* Methods:
  - 'frequent' or 'mode'
  - 'custom_value'
  - 'random'
1. **fit**(*df, cat_vars*)
2. **transform**(*df*)

## 5. Outlier Treatment

```python
from fast_ml.outlier_treatment import OutlierTreatment
```
### 5.1) class OutlierTreatment 
* Methods:
  - 'iqr' or 'IQR'
  - 'gaussian'
1. **fit**(*df, num_vars*)
2. **transform**(*df*)
  
## 6. Feature Engineering
```python
from fast_ml.feature_engineering import FeatureEngineering_Numerical, FeatureEngineering_Categorical, FeatureEngineering_DateTime
```
### 6.1) class FeatureEngineering_Numerical 
```python
from fast_ml.feature_engineering import FeatureEngineering_Categorical

num_binner = FeatureEngineering_Numerical(method = '10p', adaptive = True)

#Scikit-learn type fit() transform() functionality
# Use fit() only on the train dataset
num_binner.fit(train, num_vars)

# Use transform() on train/test dataset
train = num_binner.transform(train)
test = num_binner.transform(test)
```
* Methods:
  - '5p'  : [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
  - '10p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  - '20p' : [0, 20, 40, 60, 80, 100]
  - '25p' : [0, 25, 50, 75, 100]
  - '95p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
  - '98p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100]
  - 'custom' : Custom Buckets
1. **fit**(*df, num_vars*)
2. **transform**(*df*)

### 6.2) class FeatureEngineering_Categorical(*model=None, method='label', drop_last=False*):
```python
from fast_ml.feature_engineering import FeatureEngineering_Categorical

rare_encoder_5 = FeatureEngineering_Categorical(method = 'rare')

#Scikit-learn type fit() transform() functionality
# Use fit() only on the train dataset
rare_encoder_5.fit(train, cat_vars, rare_tol=5)

# Use transform() on train/test dataset
train = rare_encoder_5.transform(train)
test = rare_encoder_5.transform(test)
```
* Methods:
  - 'rare_encoding' or 'rare'
  - 'label' or 'integer'
  - 'count'
  - 'freq'
  - 'ordered_label'
  - 'target_ordered'
  - 'target_mean'
  - 'target_prob_ratio'
  - 'target_woe'
1. **fit**(*df, cat_vars, target=None, rare_tol=5*)
2. **transform**(*df*)
 
### 6.3) class FeatureEngineering_DateTime (drop_orig=True)
```python
from fast_ml.feature_engineering import FeatureEngineering_DateTime

dt_encoder = FeatureEngineering_DateTime()

#Scikit-learn type fit() transform() functionality
# Use fit() only on the train dataset
dt_encoder.fit(train, datetime_vars, prefix = 'default')

# Use transform() on train/test dataset
train = dt_encoder.transform(train)
test = dt_encoder.transform(test)
```
1. **fit**(*df, datetime_variables, prefix = 'default'*)
2. **transform**(*df*)


## 6. Model Evaluation
1. model_save (model, model_name)
2. model_load (model_name)
3. plot_confidence_interval_for_data (model, X)
4. plot_confidence_interval_for_variable (model, X, y, variable)


---

