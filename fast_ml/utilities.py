import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

def __printmd__(string):
    display(Markdown(string))

    
def __normality_diagnostic__ ( s):
    plt.figure(figsize = (16, 4))

    plt.subplot(1,2,1)
    sns.distplot(s, hist = True, fit = norm, kde = True)
    plt.title('Histogram')

    plt.subplot(1,2,2)
    stats.probplot(s, dist="norm", plot=plt)
    plt.ylabel('RM Quantiles')

    plt.show()


def __missing_rare_category__ (df, c, add_missing, add_rare, tol=0.05):
    length_df = len(df)
    
    if add_missing:
        df[c] = df[c].fillna('Missing')
    
    s = pd.Series(df[c].value_counts() / length_df)
    s.sort_values(ascending = False, inplace = True)
    
    if add_rare:
        non_rare_label = [ix for ix, perc in s.items() if perc>tol]
        df[c] = np.where(df[c].isin(non_rare_label), df[c], 'Rare')

    return df
        
def __plot_categories__( df, c,  add_missing = False, add_rare = False, tol=0.05):

    length_df = len(df)
    
    df = __missing_rare_category__(df, c, add_missing, add_rare, tol=0.05)

    plot_df = pd.Series(df[c].value_counts() / length_df)
    plot_df.sort_values(ascending = False, inplace = True)


    fig = plt.figure(figsize=(12,4))
    ax = plot_df.plot.bar(color = 'royalblue')
    ax.set_xlabel(c)
    ax.set_ylabel('Percentage')
    ax.axhline(y=0.05, color = 'red')
    plt.show()

    
def __plot_categories_with_target__( df, c, target):

    plot_df = __calculate_mean_target_per_category__(df, c, target)
    #plot_df.reset_index(drop = True, inplace=True)


    fig, ax = plt.subplots(figsize=(12,4))
    plt.xticks(plot_df.index, plot_df[c], rotation = 90)

    ax.bar(plot_df.index, plot_df['perc'], align = 'center', color = 'lightgrey')

    ax2 = ax.twinx()
    ax2.plot(plot_df.index, plot_df[target], color = 'green')

    ax.axhline(y=0.05, color = 'red')

    ax.set_xlabel(c)
    ax.set_ylabel('Percentage Distribution')
    ax2.set_ylabel('Mean Target Value')


    plt.show()

'''
def __calculate_mean_target_per_category__(df, c, target):
    
    length_df = len(df)
    temp = pd.DataFrame(df[c].value_counts()/length_df)
    temp = pd.concat([temp, pd.DataFrame(df.groupby(c)[target].mean())], axis=1)
    temp.columns = ['perc', target]
    temp.reset_index(inplace=True)
    temp.sort_values(by='perc', ascending = False, inplace=True)
    return temp
'''

def __calculate_mean_target_per_category__(df, c, target):
    
    length_df = len(df)
    data = {'count' : df[c].value_counts(), 'perc' : df[c].value_counts()/length_df}
    temp = pd.DataFrame(data)
    temp = pd.concat([temp, pd.DataFrame(df.groupby(c)[target].mean())], axis=1)

    temp.reset_index(inplace=True)
    temp.columns = [c, 'count', 'perc', target]
    temp.sort_values(by='perc', ascending = False, inplace=True)

    return temp

def __plot_target_with_categories__(df, c, target):
    
    fig = plt.figure(figsize=(12,6))
    for cat in df[c].unique():
        df[df[c]==cat][target].plot(kind = 'kde', label = cat)

    plt.xlabel(f'Distribution of {target}')
    plt.legend(loc='best')
    plt.show()
    

def rare_encoding(df, variables, tol = 0.05):
    for var in variables:
        s = df[var].value_counts()/len(df[var])
        non_rare_labels = [cat for cat, perc in s.items() if perc >=tol]
        df[var] = np.where(df[var].isin(non_rare_labels), df[var], 'Rare')
        
    return df