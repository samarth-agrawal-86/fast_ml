# fast_ml
## Glossary
> * df : refers to dataframe
> * variable : single variable needs to be passed Ex 'V1'
> * variables : variables need to be passed as list. Even if it is single variable it has to be passed in list format. ex. ['V1', 'V2] or ['V1']
> * target : refers to target variable 
> * model : ML problem type. use 'classification' or 'clf' for classification problems and 'regression' or 'reg' for regression problems
> * method : refers to various techniques available for Missing Value Imputation, Feature Engieering... as available in each module

## 0. Utilities

`from fast_ml.utilities import *`
1. reduce_memory_usage(df, convert_to_category = False)
    * This function reduces the memory used by dataframe
2. display_all(df)
    * Use this function to show all rows and all columns of dataframe. By default pandas only show top and bottom 20 rows, columns

## 1. Exploratory Data Analysis (EDA)

`from fast_ml import eda`

### 1.1) Overview
1. eda.df_summary(df)
    * Returns a dataframe with useful summary - variables, datatype, number of unique values, sample of unique values, missing count, missing percent

### 1.2) Numerical Variables
1. eda.numerical_variable_detail(df, variable, model = None, target=None, threshold = 20)
    * Various summary statistics, spread statistics, outlier, missing values, transformation diagnostic... a detailed analysis for a single variable provided as input
2. eda.numerical_plots(df, variables, normality_check = False)
    * Uni-variate plots - Variable Distribution of all the numerical variables provided as input with target. Can also get the Q-Q plot for assessing the normality
3. eda.numerical_plots_with_target(df, variables, target, model)
    * Bi-variate plots - Scatter plot of all the numerical variables provided as input with target.

### 1.3) Categorical Variables
1. eda.categorical_variable_detail(df, variable, model = None, target=None,  rare_tol=5)
    * Various summary statistics, missing values, distributions ... a detailed analysis for a single variable provided as input
2. eda.categorical_plots(df, variables, add_missing = True, add_rare = False, rare_tol=5)
    * Uni-variate plots - distribution of all the categorical provided as input
3. eda.categorical_plots_with_target(df, variables, target, model='clf', add_missing = True,  rare_tol = 5)
    * Bi-variate plots - distribution of all the categorical provided as input with target
4. eda.categorical_plots_with_rare_and_target(df, variables, target, model = 'clf', add_missing = True, rare_v1=5, rare_v2=10)
    * Bi-variate plots - distribution of all the categorical provided as input with target with 2 inputs as rare threshold. Useful for deciding the rare bucketing
5. eda.categorical_plots_for_miss_and_freq(df, variables, target, model = 'reg')
    * Uni-variate plots - distribution of all the categorical provided as input with target with 2 inputs as rare threshold. Useful for deciding the rare bucketing

## 2. Missing Data Analysis

`from fast_ml.missing_data_analysis import MissingDataAnalysis`
### 2.1) MissingDataAnalysis Class
1. calculate_missing_values()
2. explore_numerical_imputation (variable)
3. explore_categorical_imputation (variable)


## 3. Missing Data Imputation

`from fast_ml.missing_data_imputation import MissingDataImputer_Numerical, MissingDataImputer_Categorical`
### 3.1) MissingDataImputer_Numerical Class
* fit(df, variables)
* transform(df)

### 3.2) MissingDataImputer_Categorical Class
* fit(df, variables)
* transform(df)

## 4. Feature Engineering

`from fast_ml.feature_engineering import FeatureEngineering_Numerical, FeatureEngineering_Categorical, FeatureEngineering_DateTime`
### 4.1) FeatureEngineering_Numerical Class
* TBD

### 4.2) FeatureEngineering_Categorical Class
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
 
### 4.3) FeatureEngineering_DateTime
* FeatureEngineering_DateTime (df, datetime_var, prefix, drop_orig=True)
