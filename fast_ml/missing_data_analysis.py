# Missing Value Treatment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from fast_ml.utilities import __printmd__ , __normality_diagnostic__ , __plot_categories__ , \
__plot_categories_with_target__ , __calculate_mean_target_per_category__ , __plot_target_with_categories__

class MissingDataAnalysis:
    def __init__ (self, df, target = None, model = None):
        self.__df__ = df
        self.__target__ = target
        self.__model__ = model

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
        Compares the results from various imputation methods so that you can choose the best suited one


        # 1st chart => existing categories and avg target value
        # 2nd chart => missing value replaced by frequent category ; then plot a chart with target value
        # 3rd chart => missing value replaced by 'Missing' category ; then plot a chart with target value
        # 4th chart => missing value replaced by random distribution ; then plot a chart with target value

        """
        df = self.__df__
        c = variable

        __printmd__ ('**<u>Missing Values :</u>**')

        print ('  Number :', df[c].isnull().sum())
        print ('  Percentage :', df[c].isnull().mean()*100, '%')
        print ()

        __printmd__(f'**<u>We have following options for Imputing the Missing Value for Categorical Variable, {c} :</u>**')
        print ('  1. Imputing missing values by Frequent Category' )
        print ('  2. Imputing missing values by Missing Label' )
        print ('  3. Imputing missing values by Randomly selected value' )

        print ()
        print ("Let's visualize the impact of each imputation and compare it with original distribution")
        print ()

        __printmd__ ('**<u>1. Original Distribution of all Categories :</u>**')
        __plot_categories_with_target__(df, c, target = self.__target__)

        __printmd__ ('**<u>2. All Categories after Frequent Category Imputation :</u>**')

        # Frequent value
        print ('Look at the Distibution of Frequent Category and Missing Data. Are there some major differences')
        fig = plt.figure(figsize = (8,4))
        ax = fig.add_subplot(111)

        value = df[c].mode().item()
        print ('\n\nMost Frequent Category: ', value)

        df[df[c] == value][self.__target__].plot(kind = 'kde', ax = ax, color = 'blue')

        # NA Value
        df[df[c].isnull()][self.__target__].plot(kind = 'kde', ax = ax, color = 'red')

        # Add the legend
        labels = ['Most Frequent category', 'with NA']
        ax.legend(labels, loc = 'best')
        plt.show()


        df[c+'_freq'] = df[c].fillna(value)

        __plot_categories_with_target__(df, c+'_freq', target = self.__target__)


        print ("3. All Categories after Missing Label Imputation")
        value = 'Missing'
        df[c+'_miss'] = df[c].fillna(value)

        __plot_categories_with_target__(df, c+'_miss', target = self.__target__)


        print ("4. All Categories after Randomly Selected Value Imputation")
        temp = self.__random_category_imputation__(c)
        __plot_categories_with_target__(temp, c+'_random', target = self.__target__)




    def __random_category_imputation__(self, c):

        # Get the number of null values for variable
        number_nulls = self.__df__[c].isnull().sum()

        # Get that many number of values from dataset chosen at random
        random_sample = self.__df__[c].dropna().sample(number_nulls, random_state = 0)

        # Set the index of random sample to that of null values
        random_sample.index = self.__df__[self.__df__[c].isnull()].index

        # make a copy of dataset including NA
        self.__df__[c+'_random'] = self.__df__[c].copy()

        # replace the NA in newly created variable
        self.__df__.loc[self.__df__[c].isnull(), c+'_random'] = random_sample

        return self.__df__
