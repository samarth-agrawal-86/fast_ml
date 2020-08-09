import pandas as pd
import numpy as np




class OutlierTreatment():   
    
    def __init__ (self, method, tol = 1.5):
        """
    This functions checks for outliers in the dataset using the Inter Quartile Range (IQR) calculation & Gaussian Calculation
    
    
    Parameters:
    -----------
        method:
            'iqr' or 'IQR' for interquartile calculation based outlier treatment
            'gaussian' for gaussian (meand and std dev) calculation based outlier treatment
        tol : tolerance value(default value = 1.5) Usually it is used as 1.5 or 3
        
    Returns:
    --------
        dataframe with variables that contain outliers
        """
        self.method = method
        self.tol = tol
        
        
    def fit(self, df, variables=None):
        """
        Parameters:
        -----------
            df : dataset on which you are working on
            variables: optional parameter. list of all the numeric variables. 
                   if not provided then it automatically identifies the numeric variables and analyzes for them

        """
        tol = self.tol
        self.param_dict_ ={}
        
        if variables == None:
            variables = df.select_dtypes(exclude = ['object']).columns
            # Check for datetime variable. exclude = 'datetime', 'datetime64'
            #variables = df.select_dtypes(include = ['int', 'float']).columns
        else:
            variables = variables
        
        if self.method =='log':
            print ('Fit method not required; You can directly use the transform method')

        if self.method =='iqr' or self.method =='IQR':   
            for var in variables:
                
                s = df.loc[df[var].notnull(), var]

                quartile_1, quartile_3 = np.percentile(s, [25,75])
                iqr = quartile_3 - quartile_1
                lower_bound = quartile_1 - tol*iqr
                upper_bound = quartile_3 + tol*iqr
            
                self.param_dict_[var] = {'lower_bound': lower_bound, 'upper_bound':upper_bound}
                
            
        if self.method == 'gaussian':
            for var in variables:
                s = df.loc[df[var].notnull(), var]
                
                mean_value = np.mean(s)
                std_dev = np.std(s)
                
                lower_bound = mean_value - tol*std_dev
                upper_bound = mean_value + tol*std_dev
                
                self.param_dict_[var] = {'lower_bound': lower_bound, 'upper_bound':upper_bound}
                    
                    
        return self   
       
            
    
    def transform(self,df):
        """
        Parameters:
        -----------
            df : dataset on which you are working on
        
        Returns:
        --------
            dataframe with outlier treated values
        """
        if self.method == 'log':
            None
            
        for var, mapper in self.param_dict_.items():
            lower_bound, upper_bound = mapper['lower_bound'], mapper['upper_bound']
            df[var] = np.where(df[var]>upper_bound, upper_bound, df[var])
            df[var] = np.where(df[var]<lower_bound, lower_bound, df[var])

        return df
                   

        