import sys
import pandas as pd
import numpy as np
from re import search
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Markdown, display

from .fast_eda import __df_info__, overview, numerical_vars, categorical_vars, printmd, display_all

class generate_report:
    
    def __init__(self, df, target, model_type, cat_threshold = 40, num_vars=None, cat_vars=None, dt_vars=None, 
                 hist_bins = 20, buckets_perc = '10p', outlier_tol=1.5, rare_tol = 5):
        
        self.eda_df = df.copy()
        self.model_type_ = model_type
        self.target_= target
        self.hist_bins = hist_bins
        self.buckets_perc = buckets_perc
        self.outlier_tol = outlier_tol
        self.rare_tol = rare_tol
        self.report_title_ = None
        self.report_user_ = None    
        self.df_summary = __df_info__(self.eda_df)
        
        if num_vars:
            self.num_vars_ = num_vars
        else:
            self.num_vars_ = list(self.df_summary.query(f"num_unique_values >{cat_threshold} and data_type_grp=='Numerical'").index)
        
        if cat_vars:
            self.cat_vars_ = cat_vars
        else:
            cv = list(self.df_summary.query("data_type_grp=='Categorical'").index)
            num_cat = list(self.df_summary.query(f"num_unique_values <={cat_threshold} and data_type_grp=='Numerical'").index)
            cv.extend(num_cat)
            self.cat_vars_ = cv
        
        if dt_vars:
            self.dt_vars_ = dt_vars
        else:
            self.dt_vars_ = list(self.df_summary.query("data_type_grp=='DateTime'").index)

        self.cat_vars_high_card_ = list(self.df_summary.query("num_unique_values >200 and data_type_grp=='Categorical'").index)
    
    def show(self):
        
        if self.report_title_: title = self.report_title_
        else: title = 'Exploratory Data Analysis Report'
        printmd(f'# {title}')

        if self.report_user_: printmd(f'### Prepared by - {self.report_user_}')
        printmd(' --- ')
        
        ######################################################################################################
        
        overview(self.eda_df, self.num_vars_, self.cat_vars_, self.cat_vars_high_card_, self.dt_vars_)
                 
        ######################################################################################################
                 
        numerical_vars(self.eda_df, self.num_vars_, self.target_, self.model_type_, 
                       self.hist_bins, self.buckets_perc, self.outlier_tol)
        
        ######################################################################################################
                 
        categorical_vars(self.eda_df, self.cat_vars_, self.target_, self.model_type_, self.rare_tol)
        
        ######################################################################################################


