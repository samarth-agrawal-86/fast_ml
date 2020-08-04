# fast_ml
 
## 0. Utilities

`from fast_ml.utilities import *`
1. reduce_memory_usage
2. display_all

## 1. Exploratory Data Analysis (EDA)

`from fast_ml.eda import *`
### Overview
1. eda_summary(df)
### Numerical Variables
1. eda_numerical_plots(df, variables)
2. eda_numerical_plots_with_target(df, variables, target)
  > * variables => variables need to be passed as list. Even if it is single variable it has to be passed in list format. ex. ['V1', 'V2] or ['V1']
  > * target => target variable 
### Categorical Variables
1. eda_categorical_plots
2. eda_categorical_plots_with_target
