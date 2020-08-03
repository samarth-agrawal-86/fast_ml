import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from fast_ml.utilities import printmd , display_all, normality_diagnostic , plot_categories , \
plot_categories_with_target , calculate_mean_target_per_category , plot_target_with_categories


def eda_summary(df):
    """
    This function gives following insights about each variable -
        Datatype of that variable
        Number of unique values is inclusive of missing values if any
        Also displays the some of the unique values. (set to display upto 10 values)
        Number of missing values in that variable
        Percentage of missing values for that variable

    Parameters:
    ----------
        df : dataframe for analysis

    Returns:
    --------
        df : returns a dataframe that contains useful info for the analysis
    """
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
    return info_df
   
def eda_numerical_variable(df, variable, model = None, target=None, threshold = 20):
    """
    This provides basic EDA of the Numerical variable passed,
        - Basic Statistics like Count, Data Type, min, max, mean, median, etc., 
        - Missing Values count and missing percentages 
        - Generates distribution plots. Histogram and KDE Plots 
        - Skewness and Kurtosis
        - Q-Q plot to check Normality
        - Box plot to check the spread outliers
        - Outliers using IQR
        - Various variable transformations
    Parameter :
    ----------
        df : Dataframe for which Analysis to be performed
        variable: Pass the Numerical variable for which EDA is required
        model : type of problem - classification or regression
            For classification related analysis. use 'classification' or 'clf'
            For regression related analysis. use 'regression' or 'reg'
        target : Define the target variable, if you want to see the relationship between given list of varaible with Target variable, default None
        threshold : if number of distinct value in a series is less than this value then consider that variable for categorical EDA
    
    Return :
    -------
        Returns summary & plots of given variable
    """
    eda_df = df.copy(deep=True)
    c = variable
    s = eda_df[c]
    
    # 1. Basic Statistics

    print ('Total Number of observations : ', len(s))
    print ()

    print ('Datatype :', (s.dtype))
    print ()

    print ('Number of distinct values :', s.nunique())
    print ()
    
    if s.nunique() < threshold:
        print (f' Number of unique values in this variable is less then threshold value of {threshold}')
        print ('\n Consider using Categorical EDA')
        sys.exit()
    
    else:
        printmd ('**<u>5 Point Summary :</u>**')

        print ('  Minimum  :\t\t', s.min(), '\n  25th Percentile :\t', np.percentile(s, 25), 
               '\n  Median :\t\t', s.median(), '\n  75th Percentile :\t', np.percentile(s, 75), 
               '\n  Maximum  :\t\t', s.max())

        print ()

        # 2. Missing values

        printmd ('**<u>Missing Values :</u>**')

        print ('  Number :', s.isnull().sum())
        print ('  Percentage :', s.isnull().mean()*100, '%')

        # 3. Histogram

        printmd ('**<u>Variable distribution and Spread statistics :</u>**')

        sns.distplot(s.dropna(), hist = True, fit = norm, kde = True)
        plt.show()

        # 4. Spread Statistics

        print ('Skewness :' , s.skew())
        print ('Kurtosis :', s.kurt())
        print ()

        # 5. Q-Q plot
        printmd ('**<u>Normality Check :</u>**')
        res = stats.probplot(s.dropna(), dist = 'norm', plot = plt)
        plt.show()

        # 6. Box plot to check the spread outliers
        print ()
        printmd ('**<u>Box Plot and Visual check for Outlier  :</u>**')
        sns.boxplot(s.dropna(), orient = 'v')
        plt.show()

        # 7. Get outliers. Here distance could be a user defined parameter which defaults to 1.5

        print ()
        printmd ('**<u>Outliers (using IQR):</u>**')

        quartile_1, quartile_3 = np.percentile(s, [25,75])
        IQR = quartile_3 - quartile_1
        upper_boundary = quartile_3 + 1.5 * IQR
        lower_boundary = quartile_1 - 1.5 * IQR

        print ('  Right end outliers :', np.sum(s>upper_boundary))
        print ('  Left end outliers :', np.sum(s < lower_boundary))


        # 8. Relationship with Target Variable
        
        printmd ('**<u>Bivariate plots: Relationship with Target Variable:</u>**')
        if target :
            if model == 'classification' or model == 'clf':
                plt.figure(figsize = (16, 4))
                plt.subplot(1, 2, 1)
                sns.boxplot(x=eda_df[target], y=c, data=eda_df)
                plt.subplot(1, 2, 2)
                sns.distplot(eda_df[eda_df[target] == 1][c], hist=False, label='Target=1')
                sns.distplot(eda_df[eda_df[target] == 0][c], hist=False, label='Target=0')
                plt.show()

        
        # 9. Various Variable Transformations

        print ()
        printmd (f'**<u>Explore various transformations for {c}</u>**')
        print ()

        print ('1. Logarithmic Transformation')
        try:
            s = np.where(s == 0, 1, s)
            s_log = np.log(s)
            normality_diagnostic(s_log)
        except:
            print ("Can't compute log transformation")

        print ('2. Exponential Transformation')
        try:
            s_exp = np.exp(s)
            normality_diagnostic(s_exp)
        except:
            print ("Can't compute Exponential transformation")

        print ('3. Square Transformation')
        try:
            s_sqr = np.square(s)
            normality_diagnostic(s_sqr)
        except:
            print ("Can't compute Square transformation")

        print ('4. Square-root Transformation')
        try:
            s_sqrt = np.sqrt(s)
            normality_diagnostic(s_sqrt)
        except:
            print ("Can't compute Square-root transformation")

        print ('5. Box-Cox Transformation')
        try:
            s_boxcox, lambda_param = stats.boxcox(s)
            normality_diagnostic(s_boxcox)
            print ('Optimal Lambda for Box-Cox transformation is :', lambda_param )
            print ()
        except:
            print ("Can't compute Box-Cox transformation")

        print ('6. Yeo Johnson Transformation')
        try:
            s = s.astype('float')
            s_yeojohnson, lambda_param = stats.yeojohnson(s)
            normality_diagnostic(s_yeojohnson)
            print ('Optimal Lambda for Yeo Johnson transformation is :', lambda_param )
            print ()
        except:
            print ("Can't compute Yeo Johnson transformation")

        
def eda_numerical_plots(df, variables, normality_check = False):
    """
    This function generates the univariate plot for the all the variables in the input variables list. 
    normality check and kde plot is optional

    Parameters:
    -----------
        df : Dataframe for which Analysis to be performed
        variables : input type list. All the Numerical variables needed for plotting
        normality_check: 'True' or 'False'
            if True: then it will plot the Q-Q plot and kde plot for the variable
            if False: just plot the histogram of the variable

    Returns:
    --------
        Numerical plots for all the variables

    """
    eda_df = df.copy(deep=True)
    length_df = len(eda_df)

    if normality_check==True:
        for i, var in enumerate(variables, 1):
            
            try:
                print (f'{i}. Plot for {var}')
                s = eda_df[var]
                normality_diagnostic(s)

            except:
                print(f"Plots for variable : {var} can't be plotted")

    if normality_check==False:
        for i, var in enumerate(variables, 1):
            
            try:
                print (f'{i}. Plot for {var}')
                s = eda_df[var]
                plt.figure(figsize = (12, 4))
                sns.distplot(s, hist = True)
                plt.title('Histogram')
                plt.show()

            except:
                print(f"Plots for variable : {var} can't be plotted")





    
def eda_numerical_plots_with_target(df, variables, target, model):
    """
    This function generates the bi-variate plot for the all the variables in the input variables list with the target

    Parameters:
    -----------
        df : Dataframe for which Analysis to be performed
        variables : input type list. All the Numerical variables needed for plotting
        target : Target variable
        model : type of problem - classification or regression
                For classification related analysis. use 'classification' or 'clf'
                For regression related analysis. use 'regression' or 'reg'
        

    Returns:
    --------
        Numerical plots for all the variables

    """
    eda_df = df.copy(deep=True)
    length_df = len(eda_df)

    if model == 'classification' or model == 'clf':

        for i, var in enumerate(variables, 1):

            try:
                print (f'{i}. Plot for {var}')
                plt.figure(figsize = (16, 4))
                plt.subplot(1, 2, 1)
                sns.boxplot(x=eda_df[target], y=var, data=eda_df)
                plt.subplot(1, 2, 2)
                sns.distplot(eda_df[eda_df[target] == 1][var], hist=False, label='Target=1')
                sns.distplot(eda_df[eda_df[target] == 0][var], hist=False, label='Target=0')
                plt.show()
            except:
                print(f"Plots for variable : {var} can't be plotted")

    if model == 'regression' or model == 'reg':
        for i, var in enumerate(variables, 1):
            try:
                print (f'{i}. Plot for {var}')
                plt.figure(figsize = (10, 4))
                sns.regplot(data = eda_df, x = var, y=target)
                plt.show()
            except:
                print(f"Plots for variable : {var} can't be plotted")


        


#### -------- Categorical Variables ------- #####

def eda_categorical_plots(df, variables, add_missing = True, add_rare = False, rare_tol=5):
    """

    Parameters:
    -----------
        df : Dataframe for which Analysis to be performed
        variables : input type list. All the categorical variables needed for plotting
        add_missing : default True. if True it will replace missing values by 'Missing'
        add_rare : default False. If True it will group all the smaller categories in a 'rare' category
        rare_tol : Threshold limit (in percentage) to combine the rare occurrence categories, (rare_tol=5) i.e., less than 5% occurance categories will be grouped and forms a rare category 

    Returns:
    --------
        Category plots for all the variables

    """
    eda_df = df.copy(deep=True)
    length_df = len(eda_df)

    for i, var in enumerate(variables, 1):

        print (f'{i}. Plot for {var}')

        if add_missing:
            eda_df[var] = eda_df[var].fillna('Missing')
        
        s = pd.Series(eda_df[var].value_counts() / length_df)
        s.sort_values(ascending = False, inplace = True)
        
        if add_rare:
            non_rare_label = [ix for ix, perc in s.items() if perc>rare_tol/100]
            eda_df[var] = np.where(eda_df[var].isin(non_rare_label), eda_df[var], 'Rare')


        plot_df = pd.Series(100*eda_df[var].value_counts() / length_df)
        plot_df.sort_values(ascending = False, inplace = True)


        fig = plt.figure(figsize=(12,4))
        ax = plot_df.plot.bar(color = 'royalblue')
        ax.set_xlabel(var)
        ax.set_ylabel('Percentage')
        ax.axhline(y=rare_tol, color = 'red')
        plt.show()


def  eda_categorical_plots_with_target(df, variables, target, add_missing = True, add_rare = False, rare_tol=5):
    """
    Parameters:
    -----------
        df : Dataframe for which Analysis to be performed
        variables : input type list. All the categorical variables needed for plotting
        target : Target variable
        add_missing : default True. if True it will replace missing values by 'Missing'
        add_rare : default = False. 
            :True: If True it will group all the smaller categories in a 'rare' category
            :False: If True it will not group all the smaller categories in a 'rare' category
            :both: it will show you categorical plots - as is and after grouping all smaller categories
        rare_tol : Threshold limit (in percentage) to combine the rare occurrence categories, (rare_tol=5) i.e., less than 5% occurance categories will be grouped and forms a rare category 

    Returns:
    --------
        Category plots for all the variables
    """
    eda_df = df.copy(deep=True)
    length_df = len(eda_df)

    for i, var in enumerate(variables, 1):

        print (f'{i}. Plot for {var}')

        if add_missing:
            eda_df[var] = eda_df[var].fillna('Missing')
        
        s = pd.Series(eda_df[var].value_counts() / length_df)
        s.sort_values(ascending = False, inplace = True)
        
        if add_rare==True:
            non_rare_label = [ix for ix, perc in s.items() if perc>rare_tol/100]
            eda_df[var] = np.where(eda_df[var].isin(non_rare_label), eda_df[var], 'Rare')

        plot_df =  calculate_mean_target_per_category (eda_df, var, target)

        fig, ax = plt.subplots(figsize=(12,4))
        plt.xticks(plot_df.index, plot_df[var], rotation = 90)

        ax.bar(plot_df.index, plot_df['perc'], align = 'center', color = 'lightgrey')

        ax2 = ax.twinx()
        ax2.plot(plot_df.index, plot_df[target], color = 'green')

        ax.axhline(y=rare_tol, color = 'red')

        ax.set_xlabel(var)
        ax.set_ylabel('Percentage Distribution')
        ax2.set_ylabel('Mean Target Value')
        plt.show()
        display_all(plot_df.set_index(var).transpose())
        print()

        if add_rare =='both':
            print('Cateroical Plots after grouping into Rare Category')
            non_rare_label = [ix for ix, perc in s.items() if perc>rare_tol/100]
            eda_df[var] = np.where(eda_df[var].isin(non_rare_label), eda_df[var], 'Rare')

            plot_df =  calculate_mean_target_per_category (eda_df, var, target)

            fig, ax = plt.subplots(figsize=(12,4))
            plt.xticks(plot_df.index, plot_df[var], rotation = 90)

            ax.bar(plot_df.index, plot_df['perc'], align = 'center', color = 'lightgrey')

            ax2 = ax.twinx()
            ax2.plot(plot_df.index, plot_df[target], color = 'green')

            ax.axhline(y=rare_tol, color = 'red')

            ax.set_xlabel(var)
            ax.set_ylabel('Percentage Distribution')
            ax2.set_ylabel('Mean Target Value')
            plt.show()
            display_all(plot_df.set_index(var).transpose())
            print()


def eda_categorical_variable(df, variable, model = None, target=None,  rare_tol=5):
    """
    This function provides EDA for Categorical variable, this includes 
        - Counts
        - Cardinality, number of Categories in each Varaible
        - Missing values counts and percentages
       
    Also Category wise basic plots will be generated for the given variable 
        - Plot Categories
        - Plot Categories by including Missing Values
        - Plot categories by combining Rare label
        - Plot categories with target
        - Plot distribution of target variable for each categories (If Target Variable is passed)
   
    Parameters :
    ----------- 
        df : Dataframe for which Analysis to be performed
        variable: Pass the variable for which EDA is required
        model : type of problem - classification or regression
            For classification related analysis. use 'classification' or 'clf'
            For regression related analysis. use 'regression' or 'reg'
        target : Define the target variable, if you want to see the relationship between given list of varaible with Target variable, default None
        rare_tol : Threshold limit to combine the rare occurrence categories, (rare_tol=5) i.e., less than 5% occurance categories will be grouped and forms a rare category   
            
        
     Return :
     -------
     
     Returns summary & plots of given variable
    """
    eda_df = df.copy(deep=True)
    c = variable
    s = eda_df[c]
    
    # 1. Basic Statistics
    printmd ('**<u>Basic Info :</u>**')
    print ('Total Number of observations : ', len(s))
    print ()
    
    # 2. Cardinality
    printmd ('**<u>Cardinality of the variable :</u>**')
    print ('Number of Distinct Categories (Cardinality): ', len(s.unique()))
    if s.nunique()>100:
        print('Few of the values : ', s.unique()[0:50])
        print ()
    else:
        print ('Distinct Values : ', s.unique())
        print ()
    
    
    # 3. Missing Values

    printmd ('**<u>Missing Values :</u>**')
    
    nmiss = s.isnull().sum()
    print ('  Number :', s.isnull().sum())
    print ('  Percentage :', s.isnull().mean()*100, '%')

    # 4. Plot Categories
    
    printmd ('**<u>Category Plots :</u>**')
    plot_categories(eda_df, c)

    # 5. Plot Categories by including Missing Values
    
    if nmiss:
        printmd ('**<u>Category plot by including Missing Values**')
        plot_categories(eda_df, c, add_missing = True, add_rare = False, rare_tol = rare_tol)
        
    # 6. Plot categories by combining Rare label
    
    printmd ('**<u>Category plot by including missing (if any) and Rare labels**')
    print (f'Categories less than {rare_tol} value are clubbed in Rare label')
    plot_categories(eda_df, c, add_missing = True, add_rare = True, rare_tol = rare_tol)
    
    #7. Plot categories with target
    
    if target:
        printmd ('**<u>Category Plot and Mean Target value:</u>**')
        plot_categories_with_target(eda_df, c, target, rare_tol)
           

   #8. Plot distribution of target variable for each categories

    if target:
        if model == 'regression' or model == 'reg' :
            
            printmd ('**<u>Distribution of Target variable for all categories:</u>**')
            plot_target_with_categories(eda_df, c, target)
           
    
        
    