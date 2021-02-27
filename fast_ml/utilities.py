import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))

    
def normality_diagnostic (s, var):
    fig, ax = plt.subplots(figsize=(16,4))


    ax1 = plt.subplot(1,2,1)
    ax1 = sns.distplot(s, hist = True, fit = norm, kde = True)
    ax1.set_title('Histogram', fontsize=17)
    ax1.set_xlabel(var, fontsize=14)
    ax1.set_ylabel('Distribution', fontsize=14)

    ax2 = plt.subplot(1,2,2)
    stats.probplot(s, dist="norm", plot=plt)
    plt.title('Probability Plot', fontsize=17)
    plt.xlabel('Theoretical Quantiles', fontsize=14)
    plt.ylabel('RM Quantiles', fontsize=14)
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

def get_plot_df(eda_df, var, target):
    """
    Useful for classification type 
    """
    plot_df = pd.crosstab(eda_df[var], eda_df[target])
    #print(plot_df.index)
    plot_df = plot_df.reset_index()
    plot_df.rename(columns={0:'target_0', 1:'target_1'}, inplace=True)
    plot_df['total'] = plot_df['target_0'] + plot_df['target_1']
    plot_df['total_perc'] = 100*plot_df['total']/sum(plot_df['total'])
    plot_df['target_1_perc_overall'] = 100*plot_df['target_1']/sum(plot_df['target_1'])
    plot_df['target_1_perc_within'] = 100*plot_df['target_1']/( plot_df['target_0'] + plot_df['target_1'])
    plot_df.sort_values(by = 'total_perc', ascending = False, inplace = True, ignore_index = True)

    return plot_df

def plot_categories_overall_eventrate(plot_df, var, target, cat_order, title=None, rare_tol1 = None, rare_tol2 = None):

    if len(plot_df)>15: text_x = plot_df.index[-4]
    elif len(plot_df)>8: text_x = plot_df.index[-3]
    else: text_x = plot_df.index[-2]

    fig, ax = plt.subplots(figsize=(14,4))
    plt.xticks(plot_df.index, cat_order, rotation = 90)
    #ax.bar(plot_df.index, plot_df['total_perc'], align = 'center', color = 'lightgrey')
    ax = sns.barplot(data=plot_df, x=var, y='total_perc',order =cat_order, color ='lightgrey')

    ax2 = ax.twinx()
    ax2 = sns.pointplot(data = plot_df, x=var, y='target_1_perc_overall', order = cat_order, color='black')
    if title:
        ax.set_title(title, fontsize=17)
    else:
        ax.set_title(f'Event rate of target ({target}) across all categories of variable ({var}) Bins', fontsize=17)
    ax2.set_ylabel("Perc of Events within Category", fontsize=14)

    
    #ax.set_xlabel(var, fontsize=14)
    ax.set_ylabel('Perc of Categories', fontsize=14)
    ax2.set_ylabel("Perc of Events across all Categories", fontsize=14)
    hline1 = round(plot_df['target_1_perc_overall'].mean(),1)
    ax2.axhline(y=hline1, color = 'blue', alpha=0.4)
    
    # add text for horizontal line
    ax2.text(text_x, hline1+0.01, "Avg Event Rate (overall): "+str(hline1)+'%',
            fontdict = {'size': 8, 'color':'blue'})
    
    # add text for bar plot and point plot
    for pt in range(0, plot_df.shape[0]):
        ax.text(plot_df.index[pt]-0.04, 
                 plot_df.total_perc[pt]+0.04, 
                 str(round(plot_df.total_perc[pt],1))+'%',
                 fontdict = {'size': 8, 'color':'grey'})

        ax2.text(plot_df.index[pt]+0.05, 
                 plot_df.target_1_perc_overall[pt], 
                 str(round(plot_df.target_1_perc_overall[pt],1))+'%',
                 fontdict = {'size': 8, 'color':'black'})

    if rare_tol1:
        ax.axhline(y=rare_tol1, color = 'red', alpha=0.5)
    
        # add text for rare line
        ax.text(0, rare_tol1, "Rare Tol: "+str(rare_tol1)+'%', fontdict = {'size': 8, 'color':'red'})

    if rare_tol2:
        ax.axhline(y=rare_tol2, color = 'darkred', alpha=0.5)
    
        # add text for rare line
        ax.text(0, rare_tol2, "Rare Tol: "+str(rare_tol2)+'%', fontdict = {'size': 8, 'color':'darkred'})

    plt.show()


def plot_categories_within_eventrate(plot_df, var, target, cat_order, title = None, rare_tol1=None, rare_tol2=None):

    if len(plot_df)>15: text_x = plot_df.index[-4]
    elif len(plot_df)>8: text_x = plot_df.index[-3]
    else: text_x = plot_df.index[-2]

    fig, ax = plt.subplots(figsize=(14,4))
    plt.xticks(plot_df.index, cat_order, rotation = 90)
    #ax.bar(plot_df.index, plot_df['total_perc'], align = 'center', color = 'lightgrey')
    ax = sns.barplot(data=plot_df, x=var, y='total_perc',order =cat_order, color ='lightgrey')

    ax2 = ax.twinx()
    ax2 = sns.pointplot(data = plot_df, x=var, y='target_1_perc_within', order = cat_order, color='green')
    if title:
        ax.set_title(title, fontsize=17)
    else:
        ax.set_title(f'Event Rate of target ({target}) within each category of variable ({var}) Bins', fontsize=17)
    ax2.set_ylabel("Perc of Events within Category", fontsize=14)
    #ax.set_xlabel(var, fontsize=14)
    ax.set_ylabel('Perc of Categories', fontsize=14)
    hline2 = round(plot_df['target_1_perc_within'].mean(),1)
    ax2.axhline(y=hline2, color = 'magenta', alpha=0.4)
    
    # add text for horizontal line
    ax2.text(text_x, hline2+0.01, "Avg Event Rate (within): "+str(hline2)+'%',
            fontdict = {'size': 8, 'color':'magenta'})
    
    # add text for bar plot and point plot
    for pt in range(0, plot_df.shape[0]):
        ax.text(plot_df.index[pt]-0.04, 
                 plot_df.total_perc[pt]+0.04, 
                 str(round(plot_df.total_perc[pt],1))+'%',
                 fontdict = {'size': 8, 'color':'grey'})

        ax2.text(plot_df.index[pt]+0.05, 
                 plot_df.target_1_perc_within[pt], 
                 str(round(plot_df.target_1_perc_within[pt],1))+'%',
                 fontdict = {'size': 8, 'color':'green'})

    if rare_tol1:
        ax.axhline(y=rare_tol1, color = 'red', alpha=0.5)
    
        # add text for rare line
        ax.text(0, rare_tol1, "Rare Tol: "+str(rare_tol1)+'%', fontdict = {'size': 8, 'color':'red'})

    if rare_tol2:
        ax.axhline(y=rare_tol2, color = 'darkred', alpha=0.5)
    
        # add text for rare line
        ax.text(0, rare_tol2, "Rare Tol: "+str(rare_tol2)+'%', fontdict = {'size': 8, 'color':'darkred'})

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




def reduce_memory_usage(df, convert_to_category = False):
    """ 
        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
        
        WARNING! THIS CAN DAMAGE THE DATA 
        
        From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        
        Parameter:
        ----------
            df : dataframe which needs to be optimized
            convert_to_category : 'True' , 'False'. (default value = False) If true it will convert all 'object' type variables as category type. 

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
            if convert_to_category == True:
                df[col] = df[col].astype('category')
            else:
                None

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
