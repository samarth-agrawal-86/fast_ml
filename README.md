# Package Description

## Fast ML
fast_ml is a Python package with various functionalities inbuilt to make the life of data scientist much easier. <br>
fast_ml follow Scikit-learn type functionality with fit() and transform() methods to first learn the transforming parameters from training dataset and then transforms the training/validation/test dataset

## Glossary
> * df : refers to dataframe
> * variable : single variable needs to be passed Ex 'V1'
> * variables : variables need to be passed as list. Even if it is single variable it has to be passed in list format. ex. ['V1', 'V2] or ['V1']
> * target : refers to target variable 
> * model : ML problem type. use 'classification' or 'clf' for classification problems and 'regression' or 'reg' for regression problems
> * method : refers to various techniques available for Missing Value Imputation, Feature Engieering... as available in each module

## 0. Utilities

```python
from fast_ml.utilities import reduce_memory_usage, display_all
```
1. **reduce_memory_usage**(*df, convert_to_category = False*)
    * This function reduces the memory used by dataframe
2. **display_all**(*df*)
    * Use this function to show all rows and all columns of dataframe. By default pandas only show top and bottom 20 rows, columns

## 1. Exploratory Data Analysis (EDA)

```python
from fast_ml import eda
```

### 1.1) Overview
1. **eda.df_summary**(*df*)
    * Returns a dataframe with useful summary - variables, datatype, number of unique values, sample of unique values, missing count, missing percent

### 1.2) Numerical Variables
1. **eda.numerical_variable_detail**(*df, variable, model = None, target=None, threshold = 20*)
    * Various summary statistics, spread statistics, outlier, missing values, transformation diagnostic... a detailed analysis for a single variable provided as input
2. **eda.numerical_plots**(*df, variables, normality_check = False*)
    * Uni-variate plots - Variable Distribution of all the numerical variables provided as input with target. Can also get the Q-Q plot for assessing the normality
3. **eda.numerical_plots_with_target**(*df, variables, target, model*)
    * Bi-variate plots - Scatter plot of all the numerical variables provided as input with target.
4. **eda.numerical_check_outliers**(*df, variables=None, tol=1.5, print_vars = False*)

### 1.3) Categorical Variables
1. **eda.categorical_variable_detail**(*df, variable, model = None, target=None,  rare_tol=5*)
    * Various summary statistics, missing values, distributions ... a detailed analysis for a single variable provided as input
2. **eda.categorical_plots**(*df, variables, add_missing = True, add_rare = False, rare_tol=5*)
    * Uni-variate plots - distribution of all the categorical provided as input
3. **eda.categorical_plots_with_target**(*df, variables, target, model='clf', add_missing = True,  rare_tol = 5*)
    * Bi-variate plots - distribution of all the categorical provided as input with target
4. **eda.categorical_plots_with_rare_and_target**(*df, variables, target, model = 'clf', add_missing = True, rare_v1=5, rare_v2=10*)
    * Bi-variate plots - distribution of all the categorical provided as input with target with 2 inputs as rare threshold. Useful for deciding the rare bucketing
5. **eda.categorical_plots_for_miss_and_freq**(*df, variables, target, model = 'reg'*)
    * Uni-variate plots - distribution of all the categorical provided as input with target with 2 inputs as rare threshold. Useful for deciding the rare bucketing

## 2. Missing Data Analysis

```python
from fast_ml.missing_data_analysis import MissingDataAnalysis
```
### 2.1) Class MissingDataAnalysis 
1. explore_numerical_imputation (variable)
2. explore_categorical_imputation (variable)


## 3. Missing Data Imputation

```python
from fast_ml.missing_data_imputation import MissingDataImputer_Numerical, MissingDataImputer_Categorical
```
### 3.1) Class MissingDataImputer_Numerical 
* Methods:
  - 'mean'
  - 'meadian'
  - 'mode'
  - 'custom_value'
  - 'random'
1. fit(df, variables)
2. transform(df)

### 3.2) Class MissingDataImputer_Categorical
* Methods:
  - 'frequent' or 'mode'
  - 'custom_value'
  - 'random'
1. fit(df, variables)
2. transform(df)

## 4. Outlier Treatment

```python
from fast_ml.outlier_treatment import check_outliers, OutlierTreatment
```
### 4.1) Class OutlierTreatment 
* Methods:
  - 'iqr' or 'IQR'
  - 'gaussian'
1. fit(df, variables)
2. transform(df)
  
## 5. Feature Engineering

```python
from fast_ml.feature_engineering import FeatureEngineering_Numerical, FeatureEngineering_Categorical, FeatureEngineering_DateTime
```
### 5.1) Class FeatureEngineering_Numerical 
* TBD

### 5.2) Class FeatureEngineering_Categorical
* Methods:
  - 'rare-encoding' or 'rare'
  - 'label' or 'integer'
  - 'count'
  - 'freq'
  - 'ordered_label'
  - 'target_ordered'
  - 'target_mean'
  - 'target_prob_ratio'
  - 'target_woe'
 
### 5.3) FeatureEngineering_DateTime
* FeatureEngineering_DateTime (df, datetime_var, prefix, drop_orig=True)


## 6. Model Evaluation
1. model_save (model, model_name)
2. model_load (model_name)
3. plot_confidence_interval_for_data (model, X)
4. plot_confidence_interval_for_variable (model, X, y, variable)


---
### Installing
```python
pip install fast_ml
```

### Usage
```python
from fast_ml.from fast_ml.feature_engineering import FeatureEngineering_Categorical
import pandas as pd

df = pd.read_csv('train.csv')

```
