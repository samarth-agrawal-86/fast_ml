import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))

    
def normality_diagnostic ( s):
    plt.figure(figsize = (16, 4))

    plt.subplot(1,2,1)
    sns.distplot(s, hist = True, fit = norm, kde = True)
    plt.title('Histogram')

    plt.subplot(1,2,2)
    stats.probplot(s, dist="norm", plot=plt)
    plt.ylabel('RM Quantiles')

    plt.show()


def missing_rare_category (df, c, add_missing, add_rare, rare_tol=5):
    length_df = len(df)
    
    if add_missing:
        df[c] = df[c].fillna('Missing')
    
    s = 100*pd.Series(df[c].value_counts() / length_df)
    s.sort_values(ascending = False, inplace = True)
    
    if add_rare:
        non_rare_label = [ix for ix, perc in s.items() if perc>rare_tol]
        df[c] = np.where(df[c].isin(non_rare_label), df[c], 'Rare')

    return df
        
def  plot_categories ( df, c,  add_missing = False, add_rare = False, rare_tol=5):

    length_df = len(df)
    
    df =  missing_rare_category (df, c, add_missing, add_rare, rare_tol=5)

    plot_df = 100*pd.Series(df[c].value_counts() / length_df)
    plot_df.sort_values(ascending = False, inplace = True)


    fig = plt.figure(figsize=(12,4))
    ax = plot_df.plot.bar(color = 'royalblue')
    ax.set_xlabel(c)
    ax.set_ylabel('Percentage')
    ax.axhline(y=rare_tol, color = 'red')
    plt.show()

    
def  plot_categories_with_target ( df, c, target, rare_tol=5):

    plot_df =  calculate_mean_target_per_category (df, c, target)
    #plot_df.reset_index(drop = True, inplace=True)


    fig, ax = plt.subplots(figsize=(12,4))
    plt.xticks(plot_df.index, plot_df[c], rotation = 90)

    ax.bar(plot_df.index, plot_df['perc'], align = 'center', color = 'lightgrey')

    ax2 = ax.twinx()
    ax2.plot(plot_df.index, plot_df[target], color = 'green')

    ax.axhline(y=rare_tol, color = 'red')

    ax.set_xlabel(c)
    ax.set_ylabel('Percentage Distribution')
    ax2.set_ylabel('Mean Target Value')


    plt.show()

'''
def  calculate_mean_target_per_category (df, c, target):
    
    length_df = len(df)
    temp = pd.DataFrame(df[c].value_counts()/length_df)
    temp = pd.concat([temp, pd.DataFrame(df.groupby(c)[target].mean())], axis=1)
    temp.columns = ['perc', target]
    temp.reset_index(inplace=True)
    temp.sort_values(by='perc', ascending = False, inplace=True)
    return temp
'''

def  calculate_mean_target_per_category (df, c, target):
    
    length_df = len(df)
    data = {'count' : df[c].value_counts(), 'perc' : 100*df[c].value_counts()/length_df}
    temp = pd.DataFrame(data)
    temp = pd.concat([temp, pd.DataFrame(df.groupby(c)[target].mean())], axis=1)

    temp.reset_index(inplace=True)
    temp.columns = [c, 'count', 'perc', target]
    temp.sort_values(by='perc', ascending = False, inplace=True)

    return temp

def  plot_target_with_categories (df, c, target):
    
    fig = plt.figure(figsize=(12,6))
    for cat in df[c].unique():
        df[df[c]==cat][target].plot(kind = 'kde', label = cat)

    plt.xlabel(f'Distribution of {target}')
    plt.legend(loc='best')
    plt.show()
    

def display_all(df):
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000):
        display(df)

        
def rare_encoding(df, variables, rare_tol = 0.05):
    for var in variables:
        s = df[var].value_counts()/len(df[var])
        non_rare_labels = [cat for cat, perc in s.items() if perc >=rare_tol]
        df[var] = np.where(df[var].isin(non_rare_labels), df[var], 'Rare')
        
    return df




def reduce_memory_usage(df):
    """ 
        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
        
        WARNING! THIS CAN DAMAGE THE DATA 
        
        From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        
        Parameter:
        ----------
            df : dataframe which needs to be optimized
            
        Returns:
        --------
            df : returns the reduced dataframe
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
