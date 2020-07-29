import pandas as pd
import numpy as np
from fast_ml.utilities import rare_encoding

class FeatureEngineering_Categorical:
    def __init__(self, model=None, method='label', drop_last=True, n_frequent=None, rare_tol=0.05):
        '''
        Parameters:
        -----------
            model : default is None. Most of the encoding methods can be used for both classification and regression problems. however only 2 methods require model to be defined as 'classification' or 'clf'
            method :
                'rare-encoding' or 'rare' : for combining less frequent categories into 'Rare' label first. 
                'one-hot' : for one hot encoding
                'integer' or 'label' : converts categories into codes
                'count' : converts categories into the count of occurrences
                'freq' : converts categories into the freq of occurrences
                'ordered_label' : converts categories into codes but in the descending order of occurrences
                
                Target encoding methods
                'target_ordered' : converts categories into codes but in the descending order of mean target value
                'target_mean' : converts categories into mean target value
                'target_prob' : only for classification models, takes the ratio of target =1 / target =0
                'target_woe' : only for classification models, calculates the weight of evidence(woe) for each category and replaces the value for that
                
            drop_last : if method = 'one-hot' then use this parameter to drop last category 
            n_frequent : if method = 'one-hot' then use this parameter to get only the top n categories 
            rare_tol : Threshold limit to combine the rare occurrence categories, (rare_tol=0.05) i.e., less than 5% occurance categories will be grouped and forms a rare category 

            
        '''
        self.method = method
        self.drop_last = drop_last
        self.n_frequent = n_frequent
        self.model = model
        self.drop_last = drop_last
        self.rare_tol = rare_tol
        
    def fit(self, df, variables, target=None):
        '''
        
        Parameters:
        -----------
            df : training dataset
            variables : list of all the categorical variables
            target : target variable if any target encoding method is used
            
        '''
        self.param_dict_ = {}
        
        if self.method == 'rare_encoding' or self.method == 'rare':
            for var in variables:
                s = df[var].value_counts()/len(df[var])
                self.param_dict_[var] = s.to_dict()

        if self.method == 'one-hot' or self.method == 'onehot':
            for var in variables:
                cats = list(df[var].unique())
                if self.drop_last:
                    self.param_dict_[var] = cats[0:-1]
                else:
                    self.param_dict_[var] = cats
                    
        if self.method == 'integer' or self.method == 'label':
            for var in variables:
                self.param_dict_[var] = {cat:ix for ix, cat in enumerate(df[var].unique())}
        
        if self.method == 'count':
            for var in variables:
                self.param_dict_[var] = df[var].value_counts().to_dict()
        
        if self.method == 'freq' or self.method == 'frequency':
            for var in variables:
                self.param_dict_[var] = (df[var].value_counts()/len(df[var])).to_dict()
                
        
        if self.method == 'ordered_label':
            for var in variables:
                s = df[var].value_counts()
                self.param_dict_[var] = {cat:ix for ix, cat in enumerate(s.index)}
            
        
        # Target Encoding
        if self.method == 'target_ordered':
            for var in variables:
                s = df.groupby(var)[target].mean()
                self.param_dict_[var] = {cat:ix for ix, cat in enumerate(s.index)}
                
        
        if self.method == 'target_mean':
            for var in variables:
                s = df.groupby(var)[target].mean()
                self.param_dict_[var] = {cat:ix for cat, ix in s.items()}
        
        if (self.model =='classification' or self.model == 'clf'):
            if self.method =='target_prob_ratio':
                for var in variables:
                    try:
                        prob_df = pd.DataFrame(df.groupby(var)[target].mean())
                        prob_df.columns = ['target_1']
                        prob_df['target_0'] = 1 - prob_df['target_1']
                        prob_df['ratio'] = prob_df['target_1']/prob_df['target_0']
                        self.param_dict_[var] = prob_df['ratio'].to_dict()
                    except:
                        print(f'Error in the variable : {var}')
                    
        if (self.model =='classification' or self.model == 'clf'):
            if self.method =='target_woe':
                for var in variables:
                    try:
                        woe_df = pd.DataFrame(pd.crosstab(df[var], df[target], normalize='columns').mul(100))
                        woe_df.rename(columns={0: "Target_0_Per", 1: "Target_1_Per"},inplace=True)
                        woe_df['WOE'] = np.log(woe_df['Target_1_Per']/woe_df['Target_0_Per'])
                        self.param_dict_[var] = woe_df['WOE'].to_dict()
                    except:
                          print(f'Error in the variable : {var}')
                                         
        return None
    
    def transform(self, df):
        '''
        Parameters:
        -----------
            df = training dataset
            variables = list of all the categorical variables
            target = target variable if any target encoding method is used
            
        Returns:
        --------
            dataframe with all the variables encoded
        '''
        
        if self.method == 'rare_encoding' or self.method == 'rare':
            for var, mapper in self.param_dict_.items():
                non_rare_labels = [cat for cat, perc in mapper.items() if perc >=self.rare_tol]
                df[var] = np.where(df[var].isin(non_rare_labels), df[var], 'Rare')

        if self.method == 'one-hot' or self.method == 'onehot':
            for var, mapper in self.param_dict_.items():
                for category in mapper:
                    df[str(var) + '_' + str(category)] = np.where(df[var] ==category , 1, 0)
            
        if self.method in ('integer', 'label', 'count', 'freq', 'frequency', 'ordered_label', 'target_ordered', 'target_mean'):
            for var, mapper in self.param_dict_.items():
                df[var] = df[var].map(mapper)
        
        if (self.model =='classification' or self.model == 'clf'):
            if self.method in ('target_prob_ratio', 'target_woe'):
                for var, mapper in self.param_dict_.items():
                    df[var] = df[var].map(mapper)

        return df
                
        

def FeatureEngineering_DateTime (df, datetime_var, prefix, drop_orig=True):
    """
    This function extracts more features from datetime variable
    ['year', 'quarter', 'month', 'week','day','dayofweek','dayofyear',
              'is_month_end','is_month_start','is_quarter_end','is_quarter_start','is_year_end', 'is_year_start']
    
    Parameters:
    -----------
        df : dataset 
        datetime_var : only single datetime variable at one time
        prefix : prefix for the new variables that are going to get created 
        drop_orig : If True then original variable is dropped. Usually it's kept as True
        
    Returns:
    ---------
        df : returns a new dataframe with more engineered variables
        
    Example:
    --------
        train_new = FeatureEngineering_DateTime (df=train, datetime_var='Saledate', prefix='Saledate_', drop_orig=True )
    """
    
    features = ['year', 'quarter', 'month', 'week','day','dayofweek','dayofyear',
              'is_month_end','is_month_start','is_quarter_end','is_quarter_start','is_year_end', 'is_year_start']
    for f in features:
        df[prefix+f] = getattr(df[datetime_var].dt, f)
    
    if drop_orig:
        df.drop(datetime_var, axis=1, inplace = True)
    
    return df