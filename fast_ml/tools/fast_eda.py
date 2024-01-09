import sys
import pandas as pd
import numpy as np
from re import search
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))

def display_all(df):
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000):
        display(df)

def __df_info__(df):

    data_dict = {}
    for var in df.columns:
    
        data_dict[var] = {'data_type': df[var].dtype, 
                          'num_unique_values': df[var].nunique(),
                          'sample_unique_values' : df[var].unique()[0:10].tolist(),
                          'num_missing' : df[var].isnull().sum(),
                          'perc_missing' : 100*df[var].isnull().mean()
                         }

    info_df = pd.DataFrame(data_dict).transpose()
    info_df = info_df[['data_type', 'num_unique_values', 'sample_unique_values', 'num_missing', 'perc_missing']]
    
    def __data_type_grp__(x):
        x = str(x)
        if search("int|float", x) : return 'Numerical'
        elif search("object", x) : return 'Categorical'
        elif search("datetime", x) : return 'DateTime'
        else: return x
    
    info_df['data_type_grp'] = info_df['data_type'].apply(__data_type_grp__)
    info_df = info_df[['data_type', 'data_type_grp', 'num_unique_values', 'sample_unique_values','num_missing', 'perc_missing']]
    
    
    return info_df

def __df_cardinality_info__(df_summary):

    def __cardinality_check__(x):
        if x == 0 : return '0.  0'
        elif x >0 and x<=10 : return  '(0  -- 10]'
        elif x >10 and x<=20 : return '(10 -- 20]'
        elif x >20 and x<=30 : return '(20 -- 30]'
        elif x >30 and x<=40 : return '(30 -- 40]'
        elif x >40 and x<=50 : return '(40 -- 50]'
        elif x >50 and x<=100 : return '(90 -- 100]'
        elif x >100 and x<=200 : return '(100 -- 200]'
        elif x >200 and x<=500 : return '(200 -- 500]'
        elif x >500 and x<=1000 : return '(500 -- 1000]'
        elif x >1000: return '1000+'

    df_summary['cardinality_bin'] = df_summary['num_unique_values'].apply(__cardinality_check__)

    info_pivot1 = pd.pivot_table(data = df_summary, values = 'num_unique_values', 
                              margins=True, margins_name='Total',
                              index = 'cardinality_bin', columns = ['data_type_grp'], fill_value=0, aggfunc='count')
    display(info_pivot1)


def __df_missing_info__(df_summary):
    
    def __missing_bin__(x):
        if x ==0 : return '0'
        elif x >0 and x<=10 : return  '(0  -- 10]'
        elif x >10 and x<=20 : return '(10 -- 20]'
        elif x >20 and x<=30 : return '(20 -- 30]'
        elif x >30 and x<=40 : return '(30 -- 40]'
        elif x >40 and x<=50 : return '(40 -- 50]'
        elif x >50 and x<=60 : return '(50 -- 60]'
        elif x >60 and x<=70 : return '(60 -- 70]'
        elif x >70 and x<=80 : return '(70 -- 80]'
        elif x >80 and x<=90 : return '(80 -- 90]'
        elif x >90 and x<100 : return '(90 -- 100]'
        elif x ==100: return '100'

    df_summary['missing_bin'] = df_summary['perc_missing'].apply(__missing_bin__)

    printmd ('i. Variables WITHOUT missing values') 

    df_miss_res1 = pd.pivot_table(data = df_summary.query("missing_bin == '0'"), values = 'num_missing', 
                                  margins=True, margins_name='Total Non Missing',
                                  index = 'missing_bin', columns = ['data_type_grp'], fill_value=0, aggfunc='count')
    display_all(df_miss_res1)
    print()

    printmd ('ii. Variables WITH missing values') 

    df_miss_res2 = pd.pivot_table(data = df_summary.query("missing_bin != '0'"), values = 'num_missing', 
                                  margins=True, margins_name='Total Missing',
                                  index = 'missing_bin', columns = ['data_type_grp'], fill_value=0, aggfunc='count')
    display_all(df_miss_res2)


def overview(df,*var_lists):
    
    df_summary = __df_info__(df)

    printmd('# EDA for Overall Dataset')
    printmd(' --- ')

    ######################################################################################################

    #1 Shape
    printmd ('### 1. Shape of dataset:') 
    print(df.shape)
    print()
    print('Number of Numerical Variables detected : ',df_summary.query("data_type_grp == 'Numerical'").shape[0] )
    print('Number of Categorical Variables detected : ',df_summary.query("data_type_grp == 'Categorical'").shape[0] )
    print('Number of DateTime Variables detected : ',df_summary.query("data_type_grp == 'DateTime'").shape[0] )

    if len(var_lists) > 0:
        nv, cv, cve, dv = var_lists[0], var_lists[1], var_lists[2], var_lists[3]
    
        print()
        print('After calibration number of Variables used for Numerical EDA : ',len(nv) )
        print('After calibration number of Variables used for Categorical EDA : ', len(cv) )
        print(f'Out of {len(cv)} Categorical EDA Variables {len(cve)} have cardinality of more than 200: {cve}' )
        print('After calibration number of Variables used for DateTime EDA : ', len(dv) )

    ######################################################################################################

    #2 Size
    printmd('### 2. Size of dataset:') 
    df_size = df.memory_usage().sum()/(1024)
    if df_size>1020: 
        df_size = df_size/1024
        print(f'{round(df_size,2)} MB')
    else: print(f'{round(df_size,2)} KB')

    ######################################################################################################

    #3 Info
    printmd('### 3. Dataset info') 
    
    display_all(df_summary)

    ######################################################################################################

    #4 Cardinality Check
    printmd('### 4. Check Cardinality for all the variables in the dataset') 
    __df_cardinality_info__(df_summary)

    ######################################################################################################

    #5 Missing Value check
    printmd('### 5. Check Missing Values for all the variables in the dataset') 
    __df_missing_info__(df_summary)


def numerical_vars(df, variables, target, model_type, hist_bins = 20, buckets_perc = '10p', outlier_tol=1.5):
    
    eda_df = df.copy()
    if model_type in('reg', 'regression'):
        target_bins = np.percentile(a = eda_df[target], q=[0,20,40,60,80,100])
        eda_df['target_var_bin'] = pd.cut(eda_df[target], bins=target_bins, include_lowest=True)
        eda_df['target_var_label'] = pd.cut(eda_df[target], bins=target_bins, labels = [1,2,3,4,5], include_lowest=True)
    
    if isinstance(variables, str):
        var = variables
        printmd(f'# EDA for Variable : {var}')
        printmd(' --- ')
        __num_eda_func__(eda_df, var, target, model_type, hist_bins, buckets_perc, outlier_tol)
        
    elif isinstance(variables, list):
        if target in variables: variables.remove(target)
        printmd('# EDA For Numerical Variables')
        printmd(' --- ')
        
        for i, var in enumerate(variables):
            
            printmd(f'## {i+1}. EDA for Variable : {var}')
            __num_eda_func__(eda_df, var, target, model_type, hist_bins, buckets_perc, outlier_tol)
            
    

def __num_eda_func__(eda_df, var, target, model_type , hist_bins, buckets_perc, outlier_tol):
    
    s = eda_df[var]
    s_n = eda_df[var].dropna()
    
    if buckets_perc == '10p':
        buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    elif buckets_perc == '10p':
        buckets = [0, 20, 40, 60, 80, 100]
    
    var_bins = sorted(set(list(np.percentile(s_n, q=buckets))))
    eda_df['var_bin'] = pd.cut(s_n, bins = var_bins, include_lowest=True)
    eda_df['var_bin'] = eda_df['var_bin'].astype('object')
    eda_df['var_bin'].fillna('Missing', inplace=True)
    
    printmd("### 1. Spread Statistics")
    spread_stats = s_n.describe(percentiles = np.array(buckets)/100).to_frame().T
    spread_stats.drop(columns = ['min', 'max'], inplace=True)
    display(spread_stats)
    
    ######################################################################################################
    
    printmd("### 2. Missing Information")
    num_miss, perc_miss = s.isnull().sum(), round(s.isnull().mean()*100,0)
    print (f'Number of Missing Values: {num_miss}', )
    print (f'Percentage : {perc_miss} %')
    
    if model_type in ('reg', 'regression'):
        eda_df['Missing Flag'] = np.where(eda_df[var].isnull(),'Missing','Not Missing')
        fig, ax = plt.subplots(figsize=(14,6))
        ax = sns.histplot(eda_df, x=target, bins=hist_bins,  hue='Missing Flag', stat='density', element='poly', fill = False)
        ax.set_title(f'Distribution of {target} for {var} by Missing Flag', fontsize=17)
        ax.set_xlabel(target, fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        plt.show()
        
    elif model_type in ('clf', 'classification'):
        eda_df['Missing Flag'] = np.where(eda_df[var].isnull(),'Missing','Not Missing')
        plot_df = eda_df.groupby('Missing Flag')[target].agg(['count', 'mean'])
        plot_df.reset_index(inplace=True)
        plot_df.rename(columns={'mean': 'mean_value'}, inplace=True)
        plot_df['mean_value'] = round(100*plot_df['mean_value'], 0)

        fig, ax = plt.subplots(figsize=(14,6))
        ax = sns.barplot(data = plot_df, x='Missing Flag', y='mean_value', color='royalblue')
        ax.set_xlabel(f'Missing Flag for {var}', fontsize='14')
        ax.set_ylabel(f'Event Rate of target variable ({target}) in %age', fontsize=14)
        ax.set_title(f'Comparing the event rate of target ({target}) for Missing and Non Missing values of {var}', fontsize=17)
        plt.show()
    
        
    ######################################################################################################
    
    printmd("### 3. Histogram and Normality Check")
    try:
        fig, ax = plt.subplots(figsize=(14,6))

        ax1 = plt.subplot(1,2,1)
        ax1 = sns.histplot(eda_df, x=var, bins=hist_bins, kde = True)
        ax1.set_title('Histogram', fontsize=17)
        ax1.set_xlabel(var, fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)

        ax2 = plt.subplot(1,2,2)
        stats.probplot(s_n, dist="norm", plot=plt)
        plt.title('Probability Plot', fontsize=17)
        plt.xlabel('Theoretical Quantiles', fontsize=14)
        plt.ylabel('RM Quantiles', fontsize=14)
        plt.show()

    except:
        print(f"Plots for variable : {var} can't be plotted")
        
    ######################################################################################################
    
    printmd("### 4. Variable Distribution with Target Variable")
    
    if model_type in ('reg', 'regression'):
        fig, ax = plt.subplots(figsize=(14,6))
        ax = sns.histplot(eda_df, x=var, bins=hist_bins,  hue='target_var_label', element='poly', fill = False)
        ax.set_title(f'Histogram of {var} with Target variable divided in 5 bins', fontsize=17)
        ax.set_xlabel(var, fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        plt.show()
        
    elif model_type in ('clf', 'classification'):
        fig, ax = plt.subplots(figsize=(14,6))
        ax = sns.histplot(eda_df, x=var, bins=hist_bins,  hue=target, element='poly', fill = False)
        ax.set_title(f'Histogram of {var} with Target variable ({target})', fontsize=17)
        ax.set_xlabel(var, fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        plt.show()
    ######################################################################################################
    
    # 5. Relationship between Target variable and Numerical Variable 
    
    if model_type in ('reg', 'regression'):
        printmd("### 5. Scatter Plot between Variable and Target")
        fig, ax = plt.subplots(figsize=(14,6))
        ax = sns.regplot(data = eda_df, x=var, y=target)
        ax.set_title(f'Scatter plot between {var} and {target}', fontsize=17)
        ax.set_xlabel(var, fontsize=14)
        ax.set_ylabel(target, fontsize=14)
        plt.show()
    if model_type in ('clf', 'classification'):
        printmd("### 5. Mean value of Variable for each category of Target variable")
        plot_df = eda_df.groupby(target)[var].agg(['count', 'mean'])
        plot_df.reset_index(inplace=True)
        plot_df.rename(columns={'mean': 'mean_value'}, inplace=True)
        plot_df['mean_value'] = round(plot_df['mean_value'], 0)
        #cat_order = list(plot_df['Missing Flag'])
        fig, ax = plt.subplots(figsize=(14,6))
        ax = sns.barplot(data = plot_df, x=target, y='mean_value', color='royalblue')
        ax.set_xlabel(f'Target Variable ({target})', fontsize='14')
        ax.set_ylabel(f'Mean value of {var}', fontsize=14)
        ax.set_title(f'Mean Value of {var} for Target Variable ({target})', fontsize=17)
        plt.show()
    ######################################################################################################
    
    printmd("### 6. Variable Bins with Mean value of Target Variable")

    plot_df = eda_df.groupby('var_bin')[target].agg(['count', 'mean'])
    plot_df.rename(columns = {'mean' : 'mean_value'}, inplace=True)
    plot_df.reset_index(inplace=True)
    plot_df['perc'] = round(100*plot_df['count']/plot_df['count'].sum(),0)
    cat_order = list(plot_df.var_bin)
    
    fig, ax = plt.subplots(figsize = (14,6))
    ax1 = sns.barplot(data = plot_df, x = 'var_bin', y='perc', color = 'royalblue', order=cat_order)
    ax2 = ax1.twinx()
    ax2 = sns.pointplot(data = plot_df, x = 'var_bin', y='mean_value', linestyles='--', color='black', order=cat_order)


    ax1.set_title(f'Mean Value of Target variable ({target}) by binning {var}', fontsize=17)
    ax1.set_xlabel(f'{var} Bins', fontsize=14)
    ax1.set_ylabel("Bins Percentage", fontsize=14)
    ax2.set_ylabel(f"Mean value of {target}", fontsize=14)

    for pt in range(0, plot_df.shape[0]):
        if model_type in ('clf', 'classification'): 
            s_value = round(plot_df.mean_value[pt],2)
        else: s_value = int(plot_df.mean_value[pt])

        ax2.text(x=plot_df.index[pt]+0.08, 
                 y=plot_df.mean_value[pt]+0.0001, 
                 s=s_value, 
                 fontdict={'size':9, 'color':'black'})
        
        ax1.text(x=plot_df.index[pt]-0.12, 
                 y=plot_df.perc[pt]+0.06, 
                 s=str(int(plot_df.perc[pt]))+'%',
                 fontdict={'size':9, 'color':'royalblue'})
        
    plt.xticks(ticks = plot_df.index, labels = cat_order, rotation=45)
    plt.show()


    if model_type in ('reg', 'regression'):
        fig, ax = plt.subplots(figsize=(14,6))
        ax3 = sns.boxplot(data = eda_df, y=target, x='var_bin', order=cat_order)
        ax3.set_title(f'Distribution of Target variable ({target}) by binning {var}', fontsize=17)
        ax3.set_xlabel(f'{var} Bins', fontsize=14)
        ax3.set_ylabel(f'Distribution of {target}', fontsize=14)
        plt.xticks(ticks = np.arange(0,len(cat_order)), labels = cat_order, rotation=45)
        plt.show()
    ######################################################################################################
    
    printmd("### 7. Outlier Analysis")
    
    outlier_dict = {}
    quartile_1, quartile_3 = np.percentile(s_n, [25,75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - outlier_tol*iqr
    upper_bound = quartile_3 + outlier_tol*iqr

    lower_bound_outlier = np.sum(s_n<lower_bound)
    upper_bound_outlier = np.sum(s_n>upper_bound)
    #if lower_bound_outlier >0 or upper_bound_outlier>0:
    outlier_dict[var] = {'lower_bound_outliers': lower_bound_outlier, 
                             'upper_bound_outliers' : upper_bound_outlier,
                             'total_outliers' : lower_bound_outlier+upper_bound_outlier} 

    outlier_df = pd.DataFrame(data = outlier_dict).transpose()
    outlier_df = outlier_df.sort_values(by='total_outliers' , ascending = False)
    outlier_df['perc_outliers'] = round((outlier_df['total_outliers'] / len(eda_df)).mul(100),0)
    outlier_df = outlier_df[['lower_bound_outliers', 'upper_bound_outliers', 'total_outliers', 'perc_outliers']]
    display(outlier_df)
    
    ######################################################################################################
    
    printmd("### 8. Box Plots")
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax1 = sns.boxplot(data = eda_df, y=var, color='royalblue')
    ax1.set_title(f'Box Plot of {var}', fontsize=17)
    #ax1.set_xlabel(var, fontsize=14)
    ax1.set_ylabel(f'Distribution of {var}', fontsize=14)
    plt.show()
    
    if model_type in ('reg', 'regression'):
        fig, ax = plt.subplots(figsize=(14,6))
        ax2 = sns.boxplot(data = eda_df, y=var, x='target_var_label')
        ax2.set_title(f'Box Plot of {var} by Target Variable ({target}) Bins', fontsize=17)
        ax2.set_xlabel(f'{target} Bins', fontsize=14)
        ax2.set_ylabel(f'Distribution of {var}', fontsize=14)
        tmp_series = eda_df.groupby('target_var_label')[target].count()
        plt_ticks = list(np.arange(0,len(tmp_series.keys())))
        cat_order = list(tmp_series.keys())
        plt.xticks(ticks = plt_ticks, labels = cat_order, rotation=0)
        plt.show()

    
    if model_type in ('clf', 'classification'):
        fig, ax = plt.subplots(figsize=(14,5))
        ax2 = sns.boxplot(data = eda_df, y=var, x=target)
        ax2.set_title(f'Box Plot of {var} by Target variable ({target})', fontsize=17)
        ax2.set_xlabel(target, fontsize=14)
        ax2.set_ylabel(f'Distribution of {var}', fontsize=14)
        plt.show()
        
    ######################################################################################################
    
def categorical_vars(df, cat_vars, target, model_type, rare_tol=5):
    
    eda_df = df.copy()
    
    if isinstance(cat_vars, str):
        var = cat_vars
        printmd(f'# EDA for Variable : {var}')
        printmd(' --- ')
        __cat_eda_func__(eda_df, var, target, model_type, rare_tol)
        
    elif isinstance(cat_vars, list):
        if target in cat_vars: cat_vars.remove(target)
        printmd('# EDA For Categorical Variables')
        printmd(' --- ')
        
        for i, var in enumerate(cat_vars):
            
            printmd(f'## {i+1}. EDA for Variable : {var}')
            __cat_eda_func__(eda_df, var, target, model_type, rare_tol)
            
    

def __cat_eda_func__(eda_df, var, target, model_type, rare_tol ):
    
    
    ######################################################################################################
    
    printmd("### 1. Variable Info")
    print(f'Number of unique categories in {var}:  {eda_df[var].nunique()}')
    print(f"Let's look at some categories of {var}:")
    print(eda_df[var].unique()[0:20])
    
    ######################################################################################################
    
    printmd("### 2. Category Count Plot")
    plot_df_miss = eda_df.groupby(var)[target].agg(['count', 'mean', 'median'])
    plot_df_miss.reset_index(inplace=True)
    plot_df_miss.rename(columns = {'count':'cat_count','mean': 'mean_value', 'median': 'median_value'}, inplace=True)
    plot_df_miss['perc'] = round(100*plot_df_miss['cat_count']/plot_df_miss['cat_count'].sum(),0)
    cat_order_miss = list(plot_df_miss[var])

    fig, ax = plt.subplots(figsize = (14,6))
    ax1 = sns.barplot(data = plot_df_miss, x = var, y='perc', color = 'royalblue', order = cat_order_miss)
    ax1.set_title(f'Counts of labels for {var} As Is', fontsize=17)
    ax1.set_xlabel(f'Lables of {var}', fontsize=14)
    ax1.set_ylabel('Counts', fontsize=14)
    if rare_tol:
        ax1.axhline(y=rare_tol, color = 'red', alpha=0.5)
        # add text for rare line
        ax1.text(0, rare_tol, "Rare Tol: "+str(rare_tol)+'%', fontdict = {'size': 8, 'color':'red'})
    plt.show()
    
    ######################################################################################################
    
    printmd("### 3. Missing Information & Category Plot after adding MISSING Label")
    num_miss, perc_miss = eda_df[var].isnull().sum(), round(eda_df[var].isnull().mean()*100,0)
    print (f'Number of Missing Values: {num_miss}', )
    print (f'Percentage : {perc_miss} %')
    
    eda_df[var].fillna('MISSING', inplace=True)
    plot_df = eda_df.groupby(var)[target].agg(['count', 'mean', 'median'])
    plot_df.reset_index(inplace=True)
    plot_df.rename(columns = {'count':'cat_count','mean': 'mean_value', 'median': 'median_value'}, inplace=True)
    plot_df['perc'] = round(100*plot_df['cat_count']/plot_df['cat_count'].sum(),0)
    cat_order = list(plot_df[var])

    fig, ax = plt.subplots(figsize = (14,6))
    ax1 = sns.barplot(data = plot_df, x = var, y='perc', color = 'royalblue', order = cat_order)
    ax1.set_title(f'Counts of labels for {var} by adding MISSING Label', fontsize=17)
    ax1.set_xlabel(f'Lables of {var}', fontsize=14)
    ax1.set_ylabel('Percentage Distribution', fontsize=14)
    if rare_tol:
        ax1.axhline(y=rare_tol, color = 'red', alpha=0.5)
        # add text for rare line
        ax1.text(0, rare_tol, "Rare Tol: "+str(rare_tol)+'%', fontdict = {'size': 8, 'color':'red'})
    plt.show()
    
    ######################################################################################################
    
    printmd("### 4. Category Plot after adding RARE Label")
    
    non_rare_labels = list(plot_df.query(f"perc>{rare_tol}")[var])
    eda_df['var_with_rare'] = np.where(eda_df[var].isin(non_rare_labels), eda_df[var], 'RARE')

    plot_df_rare = eda_df.groupby('var_with_rare')[target].agg(['count', 'mean', 'median'])
    plot_df_rare.reset_index(inplace=True)
    plot_df_rare.rename(columns = {'count':'cat_count','mean': 'mean_value', 'median': 'median_value'}, inplace=True)
    plot_df_rare['perc'] = round(100*plot_df_rare['cat_count']/plot_df['cat_count'].sum(),0)
    cat_order_rare = list(plot_df_rare['var_with_rare'])

    fig, ax = plt.subplots(figsize = (14,6))
    ax1 = sns.barplot(data = plot_df_rare, x = 'var_with_rare', y='perc', color = 'purple', order = cat_order_rare)
    ax1.set_title(f'Counts of labels for {var} by adding RARE Label', fontsize=17)
    ax1.set_xlabel(f'Lables of {var}', fontsize=14)
    ax1.set_ylabel('Percentage Distribution', fontsize=14)
    plt.show()
        
    ######################################################################################################
    
    printmd("### 5. Category Plot with Target Variable")
    
    print('i. All the labels As Is')
    fig, ax = plt.subplots(figsize = (14,6))

    ax1 = sns.barplot(data = plot_df, x = var, y='perc', color = 'royalblue', order = cat_order)
    ax2 = ax1.twinx()
    ax2 = sns.pointplot(data = plot_df, x = var, y='mean_value', linestyles='--', color='black', order=cat_order)
    ax1.set_title(f'For {var} labels %age and Mean Value of Target Variable', fontsize=17)
    ax1.set_xlabel(f'Lables of {var}', fontsize=14)
    ax1.set_ylabel('Percentage Distribution', fontsize=14)
    ax2.set_ylabel(f'Mean Value for Target Variable {target}', fontsize=14)
    for pt in range(0, plot_df.shape[0]):
        ax1.text(x=plot_df.index[pt]-0.12, 
                 y=plot_df.perc[pt]+0.2, 
                 s=str(int(plot_df.perc[pt]))+'%',
                 fontdict={'size':9, 'color':'royalblue'})

        if model_type in ('clf', 'classification'): 
            s_value = round(plot_df.mean_value[pt],2)
        else: s_value = int(plot_df.mean_value[pt])

        ax2.text(x=plot_df.index[pt]+0.08, 
                 y=plot_df.mean_value[pt]+0.04, 
                 s=s_value, 
                 fontdict={'size':9, 'color':'black'})
    
    if rare_tol:
        ax1.axhline(y=rare_tol, color = 'red', alpha=0.5)
        # add text for rare line
        ax1.text(0, rare_tol, "Rare Tol: "+str(rare_tol)+'%', fontdict = {'size': 8, 'color':'red'})
    plt.show()
    
    print('ii. After adding the RARE label')
    fig, ax = plt.subplots(figsize = (14,6))

    ax1 = sns.barplot(data = plot_df_rare, x = 'var_with_rare', y='perc', color = 'purple', order = cat_order_rare)
    ax2 = ax1.twinx()
    ax2 = sns.pointplot(data = plot_df_rare, x = 'var_with_rare', y='mean_value', 
                        linestyles='--', color='black', order=cat_order_rare)
    ax1.set_title(f'For {var} labels %age with RARE and Mean Value of Target Variable', fontsize=17)
    ax1.set_xlabel(f'Lables of {var}', fontsize=14)
    ax1.set_ylabel('Percentage Distribution', fontsize=14)
    ax2.set_ylabel(f'Mean Value for Target Variable {target}', fontsize=14)
    
    for pt in range(0, plot_df_rare.shape[0]):
        ax1.text(x=plot_df_rare.index[pt]-0.12, 
                 y=plot_df_rare.perc[pt]+0.2, 
                 s=str(int(plot_df_rare.perc[pt]))+'%',
                 fontdict={'size':9, 'color':'purple'})

        if model_type in ('clf', 'classification'): 
            s_value = round(plot_df_rare.mean_value[pt],2)
        else: s_value = int(plot_df_rare.mean_value[pt])

        ax2.text(x=plot_df_rare.index[pt]+0.08, 
                 y=plot_df_rare.mean_value[pt]+0.04, 
                 s=s_value, 
                 fontdict={'size':9, 'color':'black'})
    plt.show()

    ######################################################################################################
    
    
    if model_type in ('reg', 'regression'):
        printmd("### 6. Category Plot with Target Variable Box Plots")
        
        print('i. All the labels As Is')
        fig, ax = plt.subplots(figsize=(14,6))
        ax1 = sns.boxplot(data = eda_df, y=target, x=var, palette='Blues' ,order=cat_order)
        ax1.set_title(f'Counts of labels for {var}', fontsize=17)
        ax1.set_xlabel(f'Lables of {var}', fontsize=14)
        ax1.set_ylabel(f'Distribution of Target Variable ({target})', fontsize=14)
        plt.show()
    
        print('ii. After adding the RARE label')
        fig, ax = plt.subplots(figsize=(14,6))
        ax1 = sns.boxplot(data = eda_df, y=target, x='var_with_rare', palette='Purples' ,order=cat_order_rare)
        ax1.set_title(f'Counts of labels for {var}', fontsize=17)
        ax1.set_xlabel(f'Lables of {var}', fontsize=14)
        ax1.set_ylabel(f'Distribution of Target Variable ({target})', fontsize=14)
        plt.show()
        
    ######################################################################################################
    




