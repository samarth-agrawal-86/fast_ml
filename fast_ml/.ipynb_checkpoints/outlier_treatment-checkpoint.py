import pandas as pd
import numpy as np


def check_outliers(df, variables=None, tol=1.5):
    """
    This functions checks for outliers in the dataset using the Inter Quartile Range (IQR) calculation
    IQR is defined as quartile_3 - quartile_1
    lower_bound = quartile_1 - tolerance_value * IQR
    upper_bound = quartile_3 + tolerance_value * IQR
    
    Parameters:
    -----------
        df : dataset on which you are working on
        variables: optional parameter. list of all the numeric variables. 
                   if not provided then it automatically identifies the numeric variables and analyzes for them
        tol : tolerance value(default value = 1.5) Usually it is used as 1.5 or 3
        
    Returns:
    --------
        dataframe with variables that contain outliers
    """
    
    outlier_dict = {}
    
    if variables == None:
        variables = df.select_dtypes('int', 'float').columns
    else:
        variables = variables
        
    for var in variables:
        s = df.loc[df[var].notnull(), var]
        
        quartile_1, quartile_3 = np.percentile(s, [25,75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - tol*iqr
        upper_bound = quartile_3 + tol*iqr
        
        lower_bound_outlier = np.sum(s<lower_bound)
        upper_bound_outlier = np.sum(s>upper_bound)
        if lower_bound_outlier >0 or upper_bound_outlier>0:
            outlier_dict[var] = {'lower_bound_outliers': lower_bound_outlier, 
                                 'upper_bound_outliers' : upper_bound_outlier,
                                 'total_outliers' : lower_bound_outlier+upper_bound_outlier} 
    
    outlier_df = pd.DataFrame(data = outlier_dict).transpose().sort_values(by='total_outliers' , ascending = False)
    outlier_df['perc_outliers'] = (outlier_df['total_outliers'] / len(df)).mul(100)
    
    return outlier_df