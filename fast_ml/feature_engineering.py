import pandas as pd
import numpy as np


class FeatureEngineering_Numerical:
    
    def __init__(self, method='10p', custom_buckets=None, adaptive = True, model='clf'):
        '''
        Select various binning techniques for bucketing the numerical variables

        Varous default methods can be used for binning as well as custom binns can be defined



        Parameters:
        -----------
            method : str. Default '10p'
                '5p'  : [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
                '10p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                '20p' : [0, 20, 40, 60, 80, 100]
                '25p' : [0, 25, 50, 75, 100]
                '95p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
                '98p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100]
                'custom' : then provide buckets in the 'custom_buckets'
            custom_buckets : list type. Default None
                if method = custom then custom_buckets need to be defined
            adaptive : bool. Default True
                If any numerical variable is skewed in nature then choosing adaptive will try to find the 
                next change in the percentile value and will not create the similar bins
            model : str. Default 'clf'
                'classification' or 'clf' for classification related analysis 
                'regression' or 'reg' for regression related analysis

        Attributes:
        -----------
            param_dict_ : dictionary
                Parameter dictionary stores the bins created for the provided variables

        '''
        self.method = method
        self.custom_buckets = custom_buckets
        self.adaptive = adaptive
        self.model = model
    
    def fit(self, df, variables):
        
        '''
        Parameters:
        -----------
            df : dataframe
            variables : list 
        
        Attributes:
        -----------
            param_dict_ : dictionary
                Parameter dictionary stores the bins created for the provided variables

        '''
        #fe_df = df.copy()
        df = df.copy()

        self.param_dict_ ={}
        
        if self.method == '5p':
            buckets = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]                
        elif self.method == '10p':
            buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        elif self.method == '20p':
            buckets = [0, 20, 40, 60, 80, 100]
        elif self.method == '25p':  
            buckets = [0, 25, 50, 75, 100]
        elif self.method == '95p':
            buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
        elif self.method == '98p':
            buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100]
        elif self.method == 'custom':
            if self.custom_buckets==None:
                raise ValueError("for 'custom' method provide a valid value in the 'custom_buckets' parameter")
            else:
                buckets = self.custom_buckets
        
        for var in variables:
            s = df[var].dropna()

            if self.adaptive:
                bins = sorted(set(list(np.percentile(s, buckets))))
            else:
                bins = list(np.percentile(s, buckets))

            bins[0] = -np.inf
            bins[-1] = np.inf
            
            self.param_dict_[var] = bins
            
    def transform(self, df):
        '''
        Parameters:
        -----------
            df : dataframe
        
        Returns:
        --------
            df : dataframe 
                dataframe with binned numerical variables
        '''
        df = df.copy()

        if self.method in ('5p', '10p', '20p', '25p', '95p', '98p', 'custom'):
            for var, binner in self.param_dict_.items():
                df[var] = pd.cut(df[var], bins = binner, include_lowest=True)
                df[var] = df[var].astype('object')
                df[var].fillna('Missing', inplace = True)
                
        return df

class FeatureEngineering_Categorical:
    def __init__(self, model=None, method='label', drop_last=False, n_frequent=None):
        """
        Parameters:
        -----------
            model : str. default None. 
                Most of the encoding methods can be used for both classification and regression problems. however only 2 methods require model to be defined as 'classification' or 'clf'
            method : str
                'rare_encoding' or 'rare' : for combining less frequent categories into 'Rare' label first. 
                'one-hot' or 'onehot' : for one hot encoding
                'integer' or 'label' : converts categories into codes
                'count' : converts categories into the count of occurrences
                'freq' : converts categories into the freq of occurrences
                'ordered_label' : converts categories into codes but in the descending order of occurrences
                
                Target encoding methods
                'target_ordered' : converts categories into codes but in the descending order of mean target value
                'target_mean' : converts categories into mean target value
                'target_prob_ratio' : only for classification models, takes the ratio of target =1 / target =0
                'target_woe' : only for classification models, calculates the weight of evidence(woe) for each category and replaces the value for that
                
            drop_last : bool. default False
                if method = 'one-hot' then use this parameter to drop last category 
            n_frequent : int. default None
                if method = 'one-hot' then use this parameter to get only the top n categories 
            
            
        """
        self.method = method
        self.drop_last = drop_last
        self.n_frequent = n_frequent
        self.model = model
        self.drop_last = drop_last
        
    def fit(self, df, variables, target=None, rare_tol=5):
        '''
        
        Parameters:
        -----------
            df : dataframe 
                training dataset on the parameters need to be learned
            variables : list type
                list of all the categorical variables for which encoding needs to be learned
            target : str
                target variable if any target encoding method is used
            rare_tol : int. Default 5. Range (0 to 100)
                Threshold limit to combine the rare occurrence categories, (default value of rare_tol=5)  i.e., less than 5% occurance categories will be grouped and forms a rare category 

            
        '''
        self.rare_tol = rare_tol
        self.param_dict_ = {}
        self.variables = variables
        df = df.copy()

        #Convert to 'Object' type
        for var in self.variables:
            df[var] = df[var].astype('object')
        
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
        
        df = df.copy()

        # Convert to object type
        for var in self.variables:
            df[var] = df[var].astype('object')


        if self.method == 'rare_encoding' or self.method == 'rare':
            for var, mapper in self.param_dict_.items():
                non_rare_labels = [cat for cat, perc in mapper.items() if perc >=self.rare_tol/100]
                df[var] = np.where(df[var].isin(non_rare_labels), df[var], 'Rare')

        if self.method == 'one-hot' or self.method == 'onehot':
            for var, mapper in self.param_dict_.items():
                for category in mapper:
                    df[str(var) + '_' + str(category)] = np.where(df[var] ==category , 1, 0)

                df.drop(var, axis=1, inplace=True)
            
        if self.method in ('integer', 'label', 'count', 'freq', 'frequency', 'ordered_label', 'target_ordered', 'target_mean'):
            for var, mapper in self.param_dict_.items():
                df[var] = df[var].map(mapper)
        
        if (self.model =='classification' or self.model == 'clf'):
            if self.method in ('target_prob_ratio', 'target_woe'):
                for var, mapper in self.param_dict_.items():
                    df[var] = df[var].map(mapper)

        return df
                
        

class FeatureEngineering_DateTime:

    def __init__(self, drop_orig=True):
        """
        This function extracts more features from datetime variable
        ['year', 'quarter', 'month', 'week','day','dayofweek','dayofyear',
                  'is_month_end','is_month_start','is_quarter_end','is_quarter_start','is_year_end', 'is_year_start']
        
        Parameters:
        -----------
            df : DataFrame
            drop_orig : bool, default True
                If True then original variable is dropped
            
        Returns:
        ---------
            df : DataFrame
                Returns a new dataframe with more engineered variables
            
        Example:
        --------
            dt_imputer = FeatureEngineering_DateTime (drop_orig=True)
            dt_imputer.fit(train, ['saledate'], prefix = 'saledate_')

            train = dt_imputer.transform(train)


        """
        self.drop_orig=drop_orig

    def fit(self, df, datetime_variables, prefix='default'):
        '''
        Parameters:
        -----------
            df : Dataframe
            datetime_variables : list type
            prefix : str. Default 'default'
                If kept default then variable name will be used as prefix for additional features that are getting created

        Attributes:
        -----------

        '''
        df = df.copy()

        self.datetime_variables = datetime_variables
        self.prefix = prefix
        self.param_dict_ = "No Parameter dictionary needed for this. Use transform() method"



    
    def transform(self, df):
        '''
        Parameters:
        -----------
            df : Dataframe

        Returns:
        --------
            df : Dataframe
                Transformed dataframe with all the additional features

        '''
        df = df.copy()

        for var in self.datetime_variables:


            if self.prefix =='default':
                pfx = str(var)+'_'

            features = ['year', 'quarter', 'month', 'week','day','dayofweek','dayofyear',
                        'is_month_end','is_month_start','is_quarter_end','is_quarter_start','is_year_end', 'is_year_start']
            for f in features:
                df[pfx+f] = getattr(df[var].dt, f)
            
            if self.drop_orig:
                df.drop(var, axis=1, inplace = True)
        
        return df