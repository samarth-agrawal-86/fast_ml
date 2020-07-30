# Missing Value Treatment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from fast_ml.utilities import printmd ,  normality_diagnostic  ,  plot_categories  , display_all, \
 plot_categories_with_target  ,  calculate_mean_target_per_category  ,  plot_target_with_categories 

class MissingDataAnalysis:
    def __init__ (self, df, target = None, model = None):
        """
        Analysing the best method for Imputing Missing Data  
    
        Parameters:
        -----------
            df : Dataset we are working on for Analysis.
            model : default is None. Most of the encoding methods can be used for both classification and regression problems. 
            target : target variable if any target
        """
         
        self.__miss_df__ = df.copy(deep=True)
        self.__target__ = target
        self.__model__ = model

        
        
    def calculate_missing_values(self):
        """
        dataframe with all the variables having missing values ordered by the percentage of missing_value_counts along with datatype of that particular variable
            
        Parameters: No Parameters needed.
        -----------
        
        Returns: It display dataset with missing counts/percentages.
        --------
        """

        df = self.__miss_df__
        vars_with_na = [var for var in df.columns if df[var].isnull().mean()>0]
        miss_df = pd.concat([df[vars_with_na].isnull().sum(), 
                             df[vars_with_na].isnull().mean().mul(100), 
                             df[vars_with_na].dtypes], axis=1 )
        miss_df.reset_index(inplace = True)
        miss_df.columns = ['variables', 'num_miss','perc_miss', 'dtype']
        miss_df.sort_values(by = 'perc_miss' , ascending = False, inplace = True)
        miss_df.reset_index(inplace = True, drop=True)
        
        return miss_df


    # ------------------------------------------------------#
    # Numerical Variable  #
    # ------------------------------------------------------#


    def explore_numerical_imputation (self, variable):

        return None

    # ------------------------------------------------------#
    # Categorical Variable #
    # ------------------------------------------------------#


    def explore_categorical_imputation (self, variable):
        """
        In this function you just pass each variable one-by-one to explore the various missing value treatments
        Parameters:
        -----------
            variables : single categorical variable
            
            Methods for Imputation:
                method : 
                    'Mode'
                    'Random_Imputation'
                    'Rare_Encding'
                    'constant'
                    'Frequency_Encoding'
            
        Returns:
        --------
            plots/graph Histograms, KDE, CountPlots for categorical variables depecting their distribution, counts, corelations among 
            themselves/the target before and after imputation for missing value are done via different
            methods(strategy) along with columns for imputed value.
            
        Compares the results from various imputation methods so that you can choose the best suited one
            # 1st chart => existing categories and avg target value
            # 2nd chart => missing value replaced by frequent category ; then plot a chart with target value
            # 3rd chart => missing value replaced by 'Missing' category ; then plot a chart with target value
            # 4th chart => missing value replaced by random distribution ; then plot a chart with target value
        """

            
        miss_df = self.__miss_df__
        c = variable

        
        printmd ('**<u>Missing Values :</u>**')

        n_miss = miss_df[c].isnull().sum()
        n_miss_perc = miss_df[c].isnull().mean()*100
        print ('  Number :', n_miss)
        print ('  Percentage :', n_miss_perc, '%')
        print ()

        if n_miss ==0:
            print('No Missing values... ')
            print('\n Stopping the process')
        
        else:

            printmd(f'**<u>We have following options for Imputing the Missing Value for Categorical Variable, {c} :</u>**')

            print ('  1. Imputing missing values by Frequent Category' )
            print ('  2. Imputing missing values by Missing Label' )
            print ('  3. Imputing missing values by Randomly selected value' )

            print ()
            print ("Let's visualize the impact of each imputation and compare it with original distribution")
            print ()


            printmd ('**<u>1. Original Distribution of all Categories :</u>**')
            plot_categories_with_target(miss_df, c, target = self.__target__)

            printmd ('**<u>2. All Categories after Frequent Category Imputation :</u>**')


            # Frequent value
            print ('Look at the Distibution of Frequent Category and Missing Data. Are there some major differences')

            if n_miss>10:
                fig = plt.figure(figsize = (8,4))
                ax = fig.add_subplot(111)

                value = miss_df[c].mode().item()
                print ('\n\nMost Frequent Category: ', value)

                miss_df[miss_df[c] == value][self.__target__].plot(kind = 'kde', ax = ax, color = 'blue')

                # NA Value
                miss_df[miss_df[c].isnull()][self.__target__].plot(kind = 'kde', ax = ax, color = 'red')

                # Add the legend
                labels = ['Most Frequent category', 'with NA']
                ax.legend(labels, loc = 'best')
                plt.show()
            else:
                print ('Not plotting the KDE plot because number of missing values is less than 10')


            miss_df[c+'_freq'] = miss_df[c].fillna(value)


            plot_categories_with_target(miss_df, c+'_freq', target = self.__target__)


            print ("3. All Categories after Missing Label Imputation")
            value = 'Missing'
            miss_df[c+'_miss'] = miss_df[c].fillna(value)

            plot_categories_with_target(miss_df, c+'_miss', target = self.__target__)


            print ("4. All Categories after Randomly Selected Value Imputation")
            if n_miss_perc <50:
                temp = self.__random_category_imputation__(c)
                plot_categories_with_target(temp, c+'_random', target = self.__target__)
            else:
                print('Since more than 50% Missing Value... Random value transformation is not advisable ')
        
        
        
    

    def __random_category_imputation__(self, c):

        # Get the number of null values for variable
        number_nulls = self.__miss_df__[c].isnull().sum()

        # Get that many number of values from dataset chosen at random
        random_sample = self.__miss_df__[c].dropna().sample(number_nulls, random_state = 0)

        # Set the index of random sample to that of null values
        random_sample.index = self.__miss_df__[self.__miss_df__[c].isnull()].index

        # make a copy of dataset including NA
        self.__miss_df__[c+'_random'] = self.__miss_df__[c].copy()

        # replace the NA in newly created variable
        self.__miss_df__.loc[self.__miss_df__[c].isnull(), c+'_random'] = random_sample

        return self.__miss_df__
