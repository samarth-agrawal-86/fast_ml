import sys
import pandas as pd
import numpy as np
from re import search
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from fast_ml.utilities import printmd , normality_diagnostic , plot_categories , \
plot_categories_with_target , calculate_mean_target_per_category , plot_target_with_categories, \
get_plot_df, plot_categories_overall_eventrate, plot_categories_within_eventrate



################################################################################
#                                     OVERALL
################################################################################

def df_info(df):
    """
    This function gives following insights about each variable -
        Datatype of that variable
        Number of unique values is inclusive of missing values if any
        Also displays the some of the unique values. (set to display upto 10 values)
        Number of missing values in that variable
        Percentage of missing values for that variable
        data type grp -> numerical, categorical, datetime
        missing bin -> missing perc groupped in (0 to 10), (10 to 20)... and so on
        cardinality_bin -> distinct values of variable grouped in (0 to 10), (10 to 20)... and so on

    Parameters:
    ----------
        df : dataframe for analysis

    Returns:
    --------
        df : returns 2 dataframes that contains useful info for the analysis
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
    
    def data_type_grp(x):
        x = str(x)
        if search("int|float", x) : return 'Numerical'
        elif search("object", x) : return 'Categorical'
        elif search("datetime", x) : return 'DateTime'
        else: return x
    
    info_df['data_type_grp'] = info_df['data_type'].apply(data_type_grp)
    info_df = info_df[['data_type', 'data_type_grp', 'num_unique_values', 'sample_unique_values','num_missing', 'perc_missing']]
    
    
    return info_df

def df_cardinality_info(df, raw_data = False):
    """
    In the function all the distinct values of variable is grouped into various bins and generates 2 outputs
    by data_type and data_type_grp
        

    Parameters:
    ----------
        df : dataframe for analysis
        raw_data: str, default 'False'. if summary df is passed then this flag is kept false.
            If original dataframe is passed then this flag is set to True

    Returns:
    --------
        df : returns 2 dataframes that contains Summary of variables cardinality by data type and data type groups
    """

    if raw_data: df_summary = df_info(df)
    else : df_summary = df

    def cardinality_check(x):
        if x == 0 : return '0.  0'
        elif x >0 and x<=10 : return  '1. (0  -- 10]'
        elif x >10 and x<=20 : return '2. (10 -- 20]'
        elif x >20 and x<=30 : return '3. (20 -- 30]'
        elif x >30 and x<=40 : return '4. (30 -- 40]'
        elif x >40 and x<=50 : return '5. (40 -- 50]'
        elif x >50 and x<=100 : return '6. (90 -- 100]'
        elif x >100 and x<=200 : return '7. (100 -- 200]'
        elif x >200 and x<=500 : return '8. (200 -- 500]'
        elif x >500 and x<=1000 : return '9. (500 -- 1000]'
        elif x >1000: return '99. 1000+'

    df_summary['cardinality_bin'] = df_summary['num_unique_values'].apply(cardinality_check)

    printmd ('**Check for Variables Cardinality**') 
    print()  
    printmd ('<u>1) By Data Type Groups</u>')  
    info_pivot1 = pd.pivot_table(data = df_summary, values = 'num_unique_values', 
                              margins=True, margins_name='Total',
                              index = 'cardinality_bin', columns = ['data_type_grp'], fill_value=0, aggfunc='count')
    display(info_pivot1)
    print("\n\n")
    
    printmd ('<u>2) By Data Type</u>')   
    info_pivot2 = pd.pivot_table(data = df_summary, values = 'num_unique_values', 
                              margins=True, margins_name='Total',
                              index = 'cardinality_bin', columns = ['data_type'], fill_value=0, aggfunc='count')
    display(info_pivot2)


def df_missing_info(df, raw_data = False):
    """
    In the function all the distinct values of variable is grouped into various bins and generates 2 outputs
    by data_type and data_type_grp
        

    Parameters:
    ----------
        df : dataframe for analysis
        raw_data: str, default 'False'. 
            If summary df is passed then this flag is kept false.
            If original dataframe is passed then this flag is set to True

    Returns:
    --------
        returns 2 dataframes that contains Summary of variables without missing values by data type and data type groups
        returns 2 dataframes that contains Summary of variables with missing values by data type and data type groups
    """

    if raw_data: df_summary = df_info(df)
    else : df_summary = df
    
    def missing_bin(x):
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

    df_summary['missing_bin'] = df_summary['perc_missing'].apply(missing_bin)

    printmd ('**I. Variables WITHOUT missing values**') 
    printmd ('<u>1) By Data Type Group</u>')
    df_miss_res1 = pd.pivot_table(data = df_summary.query("missing_bin == '0'"), values = 'num_missing', 
                                  margins=True, margins_name='Total Non Missing',
                                  index = 'missing_bin', columns = ['data_type_grp'], fill_value=0, aggfunc='count')
    display_all(df_miss_res1)
    print("\n")

    printmd ('<u>2) By Data Type</u>')
    df_miss_res2 = pd.pivot_table(data = df_summary.query("missing_bin == '0'"), values = 'num_missing', 
                                  margins=True, margins_name='Total Non Missing',
                                  index = 'missing_bin', columns = ['data_type'], fill_value=0, aggfunc='count')
    
    
    display_all(df_miss_res2)
    print("\n\n")

    printmd ('**II. Variables WITH missing values**') 
    printmd ('<u>1) By Data Type Group</u>')
    df_miss_res3 = pd.pivot_table(data = df_summary.query("missing_bin != '0'"), values = 'num_missing', 
                                  margins=True, margins_name='Total Missing',
                                  index = 'missing_bin', columns = ['data_type_grp'], fill_value=0, aggfunc='count')
    display_all(df_miss_res3)
    print("\n")

    printmd ('<u>2) By Data Type</u>')
    df_miss_res4 = pd.pivot_table(data = df_summary.query("missing_bin != '0'"), values = 'num_missing', 
                                  margins=True, margins_name='Total Missing',
                                  index = 'missing_bin', columns = ['data_type'], fill_value=0, aggfunc='count')
    display_all(df_miss_res4)
   

################################################################################
#                              NUMERICAL VARIABLES
################################################################################

def numerical_describe(df, variables=None, method='10p'):
    '''
    Parameters:
    -----------
        df : dataframe
        variables : list type, optional
            if nothing is passed then function will search for all the numerical type variables
        method : str, default '10p'
            '5p'  : [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
            '10p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
            
    Returns:
    --------
        df : Dataframe
            Various statistics about the input dataset will be returned
    '''
    if method == '5p':
        buckets = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    elif method == '10p':
        buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    
    if variables == None:
        variables = df.select_dtypes(exclude = ['object']).columns
        # Check for datetime variable. exclude = 'datetime', 'datetime64'
        # variables = df.select_dtypes(include = ['int', 'float']).columns
    else:
        variables = variables
    
    spread_stats = df[variables].describe(percentiles = np.array(buckets)/100).T
    spread_stats.drop(columns = ['min', 'max'], inplace=True)
    return spread_stats
    
def numerical_variable_detail(df, variable, model_type = None, target=None, threshold = 20):
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
        model_type : type of problem - classification or regression
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
            if model_type == 'classification' or model_type == 'clf':
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
            normality_diagnostic(s_log,c)
        except:
            print ("Can't compute log transformation")

        print ('2. Exponential Transformation')
        try:
            s_exp = np.exp(s)
            normality_diagnostic(s_exp,c)
        except:
            print ("Can't compute Exponential transformation")

        print ('3. Square Transformation')
        try:
            s_sqr = np.square(s)
            normality_diagnostic(s_sqr,c)
        except:
            print ("Can't compute Square transformation")

        print ('4. Square-root Transformation')
        try:
            s_sqrt = np.sqrt(s)
            normality_diagnostic(s_sqrt,c)
        except:
            print ("Can't compute Square-root transformation")

        print ('5. Box-Cox Transformation')
        try:
            s_boxcox, lambda_param = stats.boxcox(s)
            normality_diagnostic(s_boxcox,c)
            print ('Optimal Lambda for Box-Cox transformation is :', lambda_param )
            print ()
        except:
            print ("Can't compute Box-Cox transformation")

        print ('6. Yeo Johnson Transformation')
        try:
            s = s.astype('float')
            s_yeojohnson, lambda_param = stats.yeojohnson(s)
            normality_diagnostic(s_yeojohnson,c)
            print ('Optimal Lambda for Yeo Johnson transformation is :', lambda_param )
            print ()
        except:
            print ("Can't compute Yeo Johnson transformation")

        



def numerical_plots(df, variables, bins=20, normality_check = False):
    """
    This function generates the univariate plot for the all the variables in the input variables list. 
    normality check and kde plot is optional

    Parameters:
    -----------
        df : dataframe
            Dataframe for which Analysis to be performed
        variables : list type, optional
            All the Numerical variables needed for plotting
            if not provided then it automatically identifies the numeric variables and analyzes for them
        normality_check: 'True' or 'False'
            if True: then it will plot the Q-Q plot and kde plot for the variable
            if False: just plot the histogram of the variable

    Returns:
    --------
        Numerical plots for all the variables

    """
    eda_df = df.copy(deep=True)
    length_df = len(eda_df)
    if variables == None:
        #variables = df.select_dtypes(include = ['int', 'float']).columns
        variables = df.select_dtypes(exclude = ['object']).columns        
    else:
        variables = variables

    if normality_check==True:
        for i, var in enumerate(variables, 1):
            
            printmd (f'**{i}. Plot for {var}**')
            try:
                s = eda_df[var]
                fig, ax = plt.subplots(figsize=(16,4))

                ax1 = plt.subplot(1,2,1)
                aax1 = sns.histplot(eda_df, x=var, bins=bins, kde = True)
                ax1.set_title('Histogram', fontsize=17)
                ax1.set_xlabel(var, fontsize=14)
                ax1.set_ylabel('Count', fontsize=14)

                ax2 = plt.subplot(1,2,2)
                stats.probplot(s, dist="norm", plot=plt)
                plt.title('Probability Plot', fontsize=17)
                plt.xlabel('Theoretical Quantiles', fontsize=14)
                plt.ylabel('RM Quantiles', fontsize=14)
                plt.show()

            except:
                print(f"Plots for variable : {var} can't be plotted")

    if normality_check==False:
        for i, var in enumerate(variables, 1):
            
            printmd (f'**{i}. Plot for {var}**')
            try:
                s = eda_df[var]

                plt.figure(figsize = (12, 4))
                ax1 = sns.histplot(eda_df, x=var, bins=bins, kde = True)
                ax1.set_title('Histogram', fontsize=17)
                ax1.set_xlabel(var, fontsize=14)
                ax1.set_ylabel('Count', fontsize=14)
                plt.show()

            except:
                print(f"Plot for variable : {var} can't be plotted")





    
def numerical_plots_with_target(df, variables, target, model_type):
    """
    This function generates the bi-variate plot for the all the variables in the input variables list with the target

    Parameters:
    -----------
        df : dataframe
            Dataframe for which Analysis to be performed
        variables : list type, optional
            All the Numerical variables needed for plotting
            if not provided then it automatically identifies the numeric variables and analyzes for them
        target : str
            Target variable
        model_type : str, default 'clf'
            'classification' or 'clf' for classification related analysis 
            'regression' or 'reg' for regression related analysis
        

    Returns:
    --------
        Numerical plots for all the variables

    """
    eda_df = df.copy(deep=True)
    length_df = len(eda_df)
    if variables == None:
        #variables = df.select_dtypes(include = ['int', 'float']).columns
        variables = df.select_dtypes(exclude = ['object']).columns        
    else:
        variables = variables

    if target in variables: variables.remove(target)

    if model_type == 'classification' or model_type == 'clf':

        for i, var in enumerate(variables, 1):

            try:
                printmd (f'**{i}. Plot for {var}**')
                plt.figure(figsize = (16, 4))
                ax1 = plt.subplot(1, 2, 1)
                ax1 = sns.boxplot(x=eda_df[target], y=var, data=eda_df)
                ax1.set_title('Box Plot', fontsize=17)
                ax1.set_xlabel(f'Target Variable ({target})', fontsize=14)
                ax1.set_ylabel(var, fontsize=14)

                ax2 = plt.subplot(1, 2, 2)
                ax2 = sns.distplot(eda_df[eda_df[target] == 1][var], hist=False, label='Target=1')
                ax2 = sns.distplot(eda_df[eda_df[target] == 0][var], hist=False, label='Target=0')
                ax2.set_title('KDE Plot', fontsize=17)
                ax2.set_xlabel(var, fontsize=14)
                plt.show()

            except:
                print(f"Plots for variable : {var} can't be plotted")

    if model_type == 'regression' or model_type == 'reg':
        for i, var in enumerate(variables, 1):
            try:
                print (f'{i}. Plot for {var}')
                plt.figure(figsize = (10, 4))
                ax = sns.regplot(data = eda_df, x = var, y=target)
                ax.set_title(f'Scatter Plot bw variable ({var}) and target ({target})', fontsize=17)
                ax.set_xlabel(var, fontsize=14)
                ax.set_ylabel(target, fontsize=14)
                plt.show()
            except:
                print(f"Plots for variable : {var} can't be plotted")


        


def numerical_bins_with_target (df, variables, target, model_type='clf', sort_by='category', show_spread=True, create_buckets = True, method='5p', custom_buckets=None):
    """
    Plot the bins for numerical variables

    Various default methods can be used for binning. Or custom bucket can be provided


    Parameters:
    -----------
        df : dataframe
        variables : list type
        target : string
        model_type : string, default 'clf'
            'classification' or 'clf' for classification related analysis 
            'regression' or 'reg' for regression related analysis
        sorty_by: str, default 'category' {'category', 'mean', 'perc'}
            'category' : if x index to be sorted by the variable
            'mean' : if x index to be sorted by the mean value of the target variable in descending order
            'perc' : if x index to be sorted by the percentage value of the bin in descending order
        show_spread: bool, default 'False'. Useful when you want to see the spread of the variable as well
        create_buckets : bool, default True
        method : string, default '5p'
            '2p'  : [0,2,4,6,8,10,12.....90,92,94,96,98,100]
            '5p'  : [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
            '10p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            '20p' : [0, 20, 40, 60, 80, 100]
            '25p' : [0, 25, 50, 75, 100]
            '95p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
            '98p' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100]
            'custom' : Custom list
        custom_buckets : list type, default None
            Has to be provided compulsorily if 'custom' method is used 

    """

    if create_buckets:

        if method =='2p' :
            buckets = list(range(0,101,2))
        if method == '5p':
            buckets = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
        elif method == '10p':
            buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        elif method == '20p':
            buckets = [0, 20, 40, 60, 80, 100]
        elif method == '25p':
            buckets = [0, 25, 50, 75, 100]
        elif method == '95p':
            buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
        elif method == '98p':
            buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100]
        elif method == 'custom':
            buckets = custom_buckets

    
    eda_df = df.copy()
    if target in variables: variables.remove(target)

    if model_type in('clf', 'classification'):
        for i, var in enumerate(variables, 1):

            printmd (f'**{i}. Plot for {var}**')

            if create_buckets:
                s = eda_df[var].dropna()
                
                spread_stats = s.describe(percentiles = np.array(buckets)/100).to_frame().T
                
                if 50 not in buckets:
                    spread_stats.drop(columns = ['min', 'max', '50%'], inplace=True)
                else:
                    spread_stats.drop(columns = ['min', 'max'], inplace=True)
                
                if show_spread:
                    print(f'Spread Statistics for {var} :')
                    display_all(spread_stats)

                bins = sorted(set(list(np.percentile(s, buckets))))
                eda_df[var] = pd.cut(s, bins = bins, include_lowest=True)
                eda_df[var] = eda_df[var].astype('object')
                eda_df[var].fillna('Missing', inplace = True)

            plot_df = pd.crosstab(eda_df[var], eda_df[target])
            cat_order = (plot_df.index)
            plot_df = plot_df.reset_index()
            plot_df.rename(columns={0:'target_0', 1:'target_1'}, inplace=True)
            plot_df['total'] = plot_df['target_0'] + plot_df['target_1']
            plot_df['total_perc'] = 100*plot_df['total']/sum(plot_df['total'])
            plot_df['target_1_perc_overall'] = 100*plot_df['target_1']/sum(plot_df['target_1'])
            plot_df['target_1_perc_within'] = 100*plot_df['target_1']/( plot_df['target_0'] + plot_df['target_1'])

            # Graph 1
            plot_categories_overall_eventrate(plot_df, var, target, cat_order)

            # Graph 2
            plot_categories_within_eventrate(plot_df, var, target, cat_order)

    if model_type in('reg', 'regression'):
        for i, var in enumerate(variables, 1):

            printmd (f'**{i}. Plot for {var}**')

            if create_buckets:
                s = eda_df[var].dropna()
                
                spread_stats = s.describe(percentiles = np.array(buckets)/100).to_frame().T
                
                if 50 not in buckets:
                    spread_stats.drop(columns = ['min', 'max', '50%'], inplace=True)
                else:
                    spread_stats.drop(columns = ['min', 'max'], inplace=True)
                
                if show_spread:
                    print(f'Spread Statistics for {var} :')
                    display_all(spread_stats)

                bins = sorted(set(list(np.percentile(s, buckets))))
                eda_df[var] = pd.cut(s, bins = bins, include_lowest=True)
                eda_df[var] = eda_df[var].astype('object')
                eda_df[var].fillna('Missing', inplace = True)

            plot_df = eda_df.groupby(var)[target].agg(['count', 'mean'])
            plot_df.rename(columns = {'mean': 'mean_value'}, inplace=True)
            plot_df.reset_index(inplace=True)
            plot_df[var]= plot_df[var].astype('object')
            plot_df['perc'] = round(100*plot_df['count']/plot_df['count'].sum(),0)

            if sort_by == 'category':
                plot_df = plot_df
            elif sort_by == 'perc':
                plot_df.sort_values(by='perc', ascending=False, inplace=True)
            elif sort_by == 'mean':
                plot_df.sort_values(by='mean_value', ascending=False, inplace=True)

            plot_df.reset_index(drop=True, inplace=True)
            cat_order = list(plot_df[var])

            fig, ax = plt.subplots(figsize = (14,6))
            plt.xticks(ticks = plot_df.index, labels = cat_order, rotation=90)
            ax1 = sns.barplot(data = plot_df, x = var, y='perc', color = 'lightgrey')

            ax2 = ax.twinx()
            ax2 = sns.pointplot(data = plot_df, x = var, y='mean_value', color='black')

            ax1.set_xlabel(var, fontsize=14)
            ax1.set_ylabel("Bins Percentage", fontsize=14)
            ax2.set_ylabel(f"Mean value of {target}", fontsize=14)

            for pt in range(0, plot_df.shape[0]):
                ax1.text(x=plot_df.index[pt]-0.12, 
                         y=plot_df.perc[pt]+0.2, 
                         s=str(int(plot_df.perc[pt]))+'%',
                         fontdict={'size':8, 'color':'grey'})
                
                ax2.text(x=plot_df.index[pt]+0.08, 
                         y=plot_df.mean_value[pt]+0.04, 
                         s=int(plot_df.mean_value[pt]), 
                         fontdict={'size':8, 'color':'black'})
            plt.show()


def numerical_check_outliers(df, variables=None, tol=1.5, print_vars = False):
    """
    This functions checks for outliers in the dataset using the Inter Quartile Range (IQR) calculation
    IQR is defined as quartile_3 - quartile_1
    lower_bound = quartile_1 - tolerance_value * IQR
    upper_bound = quartile_3 + tolerance_value * IQR
    
    Parameters:
    -----------
        df : dataframe
            dataset on which you are working on
        variables: list type, optional 
            list of all the numeric variables. 
            if not provided then it automatically identifies the numeric variables and analyzes for them
        tol : float, default 1.5
            tolerance value(default value = 1.5) Usually it is used as 1.5 or 3
        
    Returns:
    --------
        Dataframe
            dataframe with variables that contain outliers
    """
    
    outlier_dict = {}
    
    if variables == None:
        #variables = df.select_dtypes(include = ['int', 'float']).columns
        variables = df.select_dtypes(exclude = ['object']).columns
        if print_vars:
            print(variables)
        
    else:
        variables = variables
        if print_vars:
            print(variables)
        
    for var in variables:
        s = df.loc[df[var].notnull(), var]
        
        quartile_1, quartile_3 = np.percentile(s, [25,75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - tol*iqr
        upper_bound = quartile_3 + tol*iqr
        
        lower_bound_outlier = np.sum(s<lower_bound)
        upper_bound_outlier = np.sum(s>upper_bound)
        #if lower_bound_outlier >0 or upper_bound_outlier>0:
        outlier_dict[var] = {'lower_bound_outliers': lower_bound_outlier, 
                                 'upper_bound_outliers' : upper_bound_outlier,
                                 'total_outliers' : lower_bound_outlier+upper_bound_outlier} 
    
    outlier_df = pd.DataFrame(data = outlier_dict).transpose()
    outlier_df = outlier_df.sort_values(by='total_outliers' , ascending = False)
    outlier_df['perc_outliers'] = (outlier_df['total_outliers'] / len(df)).mul(100)
    
    return outlier_df

#### -------- Categorical Variables ------- #####
#################################################

def categorical_plots(df, variables, add_missing = True, add_rare = False, rare_tol=5):
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
    if variables == None:
        variables = df.select_dtypes(include = ['object']).columns        
    else:
        variables = variables

    for i, var in enumerate(variables, 1):

        printmd (f'**<u> {i}. Plot for {var}</u>**')

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

        ax.set_title(f'Distribution of variable {var}', fontsize=17)
        #ax.set_xlabel(var, fontsize=14)
        ax.set_ylabel('Percentage', fontsize=14)
        ax.axhline(y=rare_tol, color = 'red')
        ax.axhline(y=rare_tol+5, color = 'darkred')
        plt.show()


def  categorical_plots_with_target(df, variables, target, model_type='clf', add_missing = True,  rare_tol1 = 5, rare_tol2 = 10):
    """
    Parameters:
    -----------
        df : Dataframe
            Dataframe for which Analysis to be performed
        variables : list type
            All the categorical variables needed for plotting
        target : str
            Target variable
        model_type : str, default 'clf'
            type of problem - classification or regression
                'classification' or 'clf' for classification related analysis 
                'regression' or 'reg' for regression related analysis
        add_missing : bool, default True. 
            if True it will replace missing values by 'Missing'
        rare_tol1 : int, default 5
            percentage line to demonstrate categories with very less data
        rare_tol2 : int, default 10
            percentage line to demonstrate categories with very less data
    
    Returns:
    --------
        Nothing
    
    Results:
    --------
        Category plots for all the variables
    """
    eda_df = df.copy(deep=True)
    length_df = len(eda_df)
    if variables == None:
        variables = df.select_dtypes(include = ['object']).columns        
    else:
        variables = variables

    if model_type in ('clf', 'classification'):

        for i, var in enumerate(variables, 1):

            printmd (f'**<u> {i}. Plot for {var}</u>**')

            if add_missing:
                eda_df[var] = eda_df[var].fillna('Missing')


            plot_df = get_plot_df(eda_df, var, target)
            cat_order = list(plot_df[var])

            # Graph:1 to show the overall event rate across categories
            plot_categories_overall_eventrate(plot_df, var, target, cat_order, rare_tol1 = rare_tol1, rare_tol2 = rare_tol2)


            # Graph:2 to show the mean target value within each category
            plot_categories_within_eventrate(plot_df, var, target, cat_order, rare_tol1 = rare_tol1, rare_tol2 = rare_tol2)

    if model_type in ('reg', 'regression'):

        for i, var in enumerate(variables, 1):

            printmd (f'**<u> {i}. Plot for {var}</u>**')

            if add_missing:
                eda_df[var] = eda_df[var].fillna('Missing')

            plot_df =  calculate_mean_target_per_category (eda_df, var, target)
            cat_order = list(plot_df[var])

            fig, ax = plt.subplots(figsize=(12,4))
            plt.xticks(plot_df.index, plot_df[var], rotation = 90)

            ax.bar(plot_df.index, plot_df['perc'], align = 'center', color = 'lightgrey')

            ax2 = ax.twinx()
            ax2 = sns.pointplot(data = plot_df, x=var, y=target, order = cat_order, color='green')

            if rare_tol1:
                ax.axhline(y=rare_tol1, color = 'red', alpha=0.5)
    
                # add text for rare line
                ax.text(0, rare_tol1, "Rare Tol: "+str(rare_tol1)+'%', fontdict = {'size': 8, 'color':'red'})

            if rare_tol2:
                ax.axhline(y=rare_tol2, color = 'darkred', alpha=0.5)
    
                # add text for rare line
                ax.text(0, rare_tol2, "Rare Tol: "+str(rare_tol2)+'%', fontdict = {'size': 8, 'color':'darkred'})


            ax.set_title(f'Mean value of target ({target}) within each category of variable ({var})', fontsize=17)
            ax2.set_ylabel('Mean Target Value', fontsize=14) 
            #ax.set_xlabel(var, fontsize=14)
            ax.set_ylabel('Perc of Categories', fontsize=14)

            plt.show()

            #display_all(plot_df.set_index(var).transpose())

        


def  categorical_plots_with_rare_and_target(df, variables, target, event_rate = 'overall', model_type = 'clf', add_missing = True, rare_tol1=5, rare_tol2=10):
    """
    Useful for deciding what percentage of rare encoding would be useful
    Parameters:
    -----------
        df : Dataframe for which Analysis to be performed
        variables : input type list. All the categorical variables needed for plotting
        target : Target variable
        event_rate : 'str', Default 'overall'
            'overall' - if overall event rate needs to be visualized
            'within' - if within category event rate needs to be visualized
        model_type : type of problem - classification or regression
                For classification related analysis. use 'classification' or 'clf'
                For regression related analysis. use 'regression' or 'reg'
        add_missing : default True. if True it will replace missing values by 'Missing'
        rare_tol1 : Input percentage as number ex 5, 10 etc (default : 5) combines categories less than that and show distribution
        rare_tol2 : Input percentage as number ex 5, 10 etc (default : 10) combines categories less than that and show distribution
       
    Returns:
    --------
        Category plots for all the variables
    """
    eda_df = df.copy(deep=True)
    length_df = len(eda_df)
    if variables == None:
        variables = df.select_dtypes(include = ['object']).columns        
    else:
        variables = variables


    if model_type in ('clf', 'classification'):
        for i, var in enumerate(variables, 1):

            print (f'{i}. Plot for {var}')

            if add_missing:
                eda_df[var] = eda_df[var].fillna('Missing')

            
            # 1st plot for categories as in in the dataset  
            plot_df = get_plot_df(eda_df, var, target)
            cat_order = list(plot_df[var])

            
            title1 = f'As Is Distribution of {var}'
            if event_rate == 'within':
                plot_categories_within_eventrate(plot_df, var, target, cat_order, title = title1, rare_tol1 = rare_tol1, rare_tol2 = rare_tol2)

            if event_rate == 'overall':
                plot_categories_overall_eventrate(plot_df, var, target, cat_order, title = title1, rare_tol1 = rare_tol1, rare_tol2 = rare_tol2)


            # 2nd plot after combining categories less than 5% 

            if rare_tol1:
                rare_tol1_df = eda_df.copy()[[var, target]]
                s_v1 = pd.Series(rare_tol1_df[var].value_counts() / length_df)
                s_v1.sort_values(ascending = False, inplace = True)
                non_rare_label = [ix for ix, perc in s_v1.items() if perc>rare_tol1/100]
                rare_tol1_df[var] = np.where(rare_tol1_df[var].isin(non_rare_label), rare_tol1_df[var], 'Rare')

                plot_df_tol1 =  get_plot_df (rare_tol1_df, var, target)
                cat_order = list(plot_df_tol1[var])

                title2 = f'Distribution of {var} after combining categories less than {rare_tol1}%'
                if event_rate == 'within':
                    plot_categories_within_eventrate(plot_df_tol1, var, target, cat_order, title = title2, rare_tol1 = rare_tol1, rare_tol2 = rare_tol2)

                if event_rate == 'overall':
                    plot_categories_overall_eventrate(plot_df_tol1, var, target, cat_order, title = title2, rare_tol1 = rare_tol1, rare_tol2 = rare_tol2)


            # 3nd plot after combining categories less than 10% 

            if rare_tol2:
                rare_tol2_df = eda_df.copy()[[var, target]]
                s_v2 = pd.Series(rare_tol2_df[var].value_counts() / length_df)
                s_v2.sort_values(ascending = False, inplace = True)
                non_rare_label = [ix for ix, perc in s_v2.items() if perc>rare_tol2/100]
                rare_tol2_df[var] = np.where(rare_tol2_df[var].isin(non_rare_label), rare_tol2_df[var], 'Rare')

                plot_df_tol2 =  get_plot_df (rare_tol2_df, var, target)
                cat_order = list(plot_df_tol2[var])

                title3 = f'Distribution of {var} after combining categories less than {rare_tol2}%'
                if event_rate == 'within':
                    plot_categories_within_eventrate(plot_df_tol2, var, target, cat_order, title = title3, rare_tol1 = rare_tol1, rare_tol2 = rare_tol2)

                if event_rate == 'overall':
                    plot_categories_overall_eventrate(plot_df_tol2, var, target, cat_order, title = title3, rare_tol1 = rare_tol1, rare_tol2 = rare_tol2)




    if model_type in ('reg', 'regression'):

        for i, var in enumerate(variables, 1):

            print (f'{i}. Plot for {var}')

            if add_missing:
                eda_df[var] = eda_df[var].fillna('Missing')

            
            # 1st plot for categories as in in the dataset        

            plot_df =  calculate_mean_target_per_category (eda_df, var, target)
            cat_order = list(plot_df[var])

            if model_type in('clf' or 'classification'):
                plot_df[target] = 100*plot_df[target]


            fig, ax = plt.subplots(figsize=(12,4))
            plt.xticks(plot_df.index, plot_df[var], rotation = 90)

            ax.bar(plot_df.index, plot_df['perc'], align = 'center', color = 'lightgrey')

            ax2 = ax.twinx()
            ax2 = sns.pointplot(data = plot_df, x=var, y=target, order = cat_order, color='green')

            ax.set_title(f'As Is Distribution of {var}', fontsize=17)
            #ax.set_xlabel(var, fontsize=14)
            ax.set_ylabel('Perc of Categories', fontsize=14)

            ax.axhline(y=rare_tol1, color = 'red')
            ax.axhline(y=rare_tol2, color = 'darkred')
            ax2.set_ylabel('Mean Target Value', fontsize=14) 

            plt.show()
            display_all(plot_df.set_index(var).transpose())
            print()

            # 2nd plot after combining categories less than 5%    
            if rare_tol1:
                rare_tol1_df = eda_df.copy()[[var, target]]
                s_v1 = pd.Series(rare_tol1_df[var].value_counts() / length_df)
                s_v1.sort_values(ascending = False, inplace = True)
                non_rare_label = [ix for ix, perc in s_v1.items() if perc>rare_tol1/100]
                rare_tol1_df[var] = np.where(rare_tol1_df[var].isin(non_rare_label), rare_tol1_df[var], 'Rare')

                plot_df_v1 =  calculate_mean_target_per_category (rare_tol1_df, var, target)
                cat_order = list(plot_df_v1[var])

                if model_type in('clf' or 'classification'):
                    plot_df_v1[target] = 100*plot_df_v1[target]

                fig, ax = plt.subplots(figsize=(12,4))
                plt.xticks(plot_df_v1.index, plot_df_v1[var], rotation = 90)

                ax.bar(plot_df_v1.index, plot_df_v1['perc'], align = 'center', color = 'lightgrey')

                ax2 = ax.twinx()
                ax2 = sns.pointplot(data = plot_df_v1, x=var, y=target, order = cat_order, color='green')

                ax.set_title(f'Distribution of {var} after combining categories less than {rare_tol1}%', fontsize=17)
                #ax.set_xlabel(var, fontsize=14)
                ax.set_ylabel('Perc of Categories', fontsize=14)

                ax.axhline(y=rare_tol1, color = 'red')
                ax.axhline(y=rare_tol2, color = 'darkred')
                ax2.set_ylabel('Mean Target Value', fontsize=14) 

                plt.show()
                display_all(plot_df_v1.set_index(var).transpose())
                print()


        # 3rd plot after combining categories less than 10%  
        if rare_tol2:
            rare_tol2_df = eda_df.copy()[[var, target]]
            s_v2 = pd.Series(rare_tol2_df[var].value_counts() / length_df)
            s_v2.sort_values(ascending = False, inplace = True)
            non_rare_label = [ix for ix, perc in s_v2.items() if perc>rare_tol2/100]
            rare_tol2_df[var] = np.where(rare_tol2_df[var].isin(non_rare_label), rare_tol2_df[var], 'Rare')

            plot_df_v2 =  calculate_mean_target_per_category (rare_tol2_df, var, target)
            cat_order = list(plot_df_v2[var])

            if model_type in('clf' or 'classification'):
                plot_df_v2[target] = 100*plot_df_v2[target]

            fig, ax = plt.subplots(figsize=(12,4))
            plt.xticks(plot_df_v2.index, plot_df_v2[var], rotation = 90)

            ax.bar(plot_df_v2.index, plot_df_v2['perc'], align = 'center', color = 'lightgrey')

            ax2 = ax.twinx()
            ax2 = sns.pointplot(data = plot_df_v2, x=var, y=target, order = cat_order, color='green')

            ax.set_title(f'Distribution of {var} after combining categories less than {rare_tol2}%', fontsize=17)
            #ax.set_xlabel(var, fontsize=14)
            ax.set_ylabel('Perc of Categories', fontsize=14)

            ax.axhline(y=rare_tol1, color = 'red')
            ax.axhline(y=rare_tol2, color = 'darkred')
            ax2.set_ylabel('Mean Target Value', fontsize=14) 

            plt.show()
            display_all(plot_df_v2.set_index(var).transpose())



def  categorical_plots_for_miss_and_freq(df, variables, target, model_type = 'reg'):
    """
    Useful ONLY for Regression Model
    Plots the KDE to check whether frequent category imputation will be suitable

    Parameters:
    -----------
        df : Dataframe for which Analysis to be performed
        variables : input type list. All the categorical variables needed for plotting
        target : Target variable
        model_type : type of problem - classification or regression
                For classification related analysis. use 'classification' or 'clf'
                For regression related analysis. use 'regression' or 'reg'
        
    Returns:
    --------
        Missing Values for  variable
        Frequent Category for variable
        KDE plot between missing values and frequent category
        
    """
    miss_df = df.copy()
    if variables == None:
        variables = df.select_dtypes(include = ['object']).columns        
    else:
        variables = variables

    if model_type in ('reg' , 'regression'):
        for i, var in enumerate(variables, 1):

            print (f'{i}. Plot for {var}')

            print ('Missing Values:')
            n_miss = miss_df[var].isnull().sum()
            n_miss_perc = miss_df[var].isnull().mean()*100
            print ('  Number :', n_miss)
            print ('  Percentage : {:1.2f}%'.format(n_miss_perc))
            
            if n_miss>10:
                fig = plt.figure(figsize = (12,4))
                ax = fig.add_subplot(111)

                value = miss_df[var].mode()
                # Careful : because some variable can have multiple modes
                if len(value) ==1:
                    print ('\n\nMost Frequent Category: ', value[0])
                    value = value[0]
                else:
                    raise ValueError(f'Variable {var} contains multiple frequent categories :', value)

                

                # Frequent Category
                miss_df[miss_df[var] == value][target].plot(kind = 'kde', ax = ax, color = 'blue')

                # NA Value
                miss_df[miss_df[var].isnull()][target].plot(kind = 'kde', ax = ax, color = 'red')

                # Add the legend
                labels = ['Most Frequent category', 'with NA']
                ax.legend(labels, loc = 'best')
                ax.set_title('KDE Plot for Frequent Category & Missing Values', fontsize=17)
                ax.set_xlabel('Distribution' , fontsize=14)
                ax.set_ylabel('Density', fontsize=14)
                plt.show()
            else:
                print ('Not plotting the KDE plot because number of missing values is less than 10')
                print()

    else:
        print ('ONLY suitable for Regression Models')


def categorical_variable_detail(df, variable, model_type = None, target=None,  rare_tol=5):
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
        model_type : type of problem - classification or regression
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
        if model_type == 'regression' or model_type == 'reg' :
            
            printmd ('**<u>Distribution of Target variable for all categories:</u>**')
            plot_target_with_categories(eda_df, c, target)
           
    
        
    