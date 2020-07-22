import pandas as pd
import numpy as np
from fast_ml.utilities import rare_encoding

class FeatureEngineering_Categorical:
    def __init__(self, model=None, method='label', drop_last=True, n_frequent=None):
        '''
        Parameters:
        -----------
            model = default is None. Most of the encoding methods can be used for both classification and regression problems. however only 2 methods require model to be defined as 'classification' or 'clf'
            method = 
                'one-hot' : for one hot encoding
                'integer' or 'label' : converts categories into codes
                'count' : converts categories into the count of occurrences
                'freq' : converts categories into the freq of occurrences
                'ordered-label' : converts categories into codes but in the descending order of occurrences
                
                Target encoding methods
                'target-ordered' : converts categories into codes but in the descending order of mean target value
                'target-mean' : converts categories into mean target value
                'target-prob' : only for classification models, takes the ratio of target =1 / target =0
                'target-woe' : only for classification models, calculates the weight of evidence(woe) for each category and replaces the value for that
                
            drop_last = if method = 'one-hot' then use this parameter to drop last category 
            n_frequent = if method = 'one-hot' then use this parameter to get only the top n categories 
            
        '''
        self.method = method
        self.drop_last = drop_last
        self.n_frequent = n_frequent
        self.model = model
        self.drop_last = drop_last
        
    def fit(self, df, variables, target=None):
        '''
        
        Parameters:
        -----------
            df = training dataset
            variables = list of all the categorical variables
            target = target variable if any target encoding method is used
            
        '''
        self.param_dict_ = {}
        
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
                self.param_dict_[var] = {cat:ix for ix, cat in s.items()}
        
        if (self.model =='classification' or self.model == 'clf'):
            if self.method =='target_prob_ratio':
                for var in variables:
                    prob_df = pd.DataFrame(df.groupby(var)[target].mean())
                    prob_df.columns = ['target_1']
                    prob_df['target_0'] = 1 - prob_df['target_1']
                    prob_df['ratio'] = prob_df['target_1']/prob_df['target_0']
                    self.param_dict_[var] = prob_df['ratio'].to_dict()
                    
        if (self.model =='classification' or self.model == 'clf'):
            if self.method =='target_woe':
                for var in variables:
                    woe_df = pd.DataFrame(pd.crosstab(df[var], df[target], normalize='columns').mul(100))
                    woe_df.rename(columns={0: "Target_0_Per", 1: "Target_1_Per"},inplace=True)
                    woe_df['WOE'] = np.log(woe_df['Target_1_Per']/woe_df['Target_0_Per'])
                    self.param_dict_[var] = woe_df['WOE'].to_dict()  
                                         
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
        if self.method == 'one-hot' or self.method == 'onehot':
            for var, mapper in self.param_dict_:
                for category in mapper:
                    df[str(var) + '_' + str(category)] = np.where(df[var] ==category , 1, 0)
            
        else:
            for var, mapper in self.param_dict_.items():
                df[var] = df[var].map(mapper)
                
        return df
                