import os
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from fast_ml.utilities import printmd 
import time
import tqdm
from multiprocessing import Process, Pool, Queue
from concurrent import futures
import pandas as pd

import sklearn
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import neural_network
from sklearn import tree
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

from sklearn.model_selection import train_test_split

from fast_ml.model_evaluation import threshold_evaluation, metrics_evaluation

import warnings
warnings.filterwarnings("ignore")

def train_valid_test_split(df, target, train_size=0.8, valid_size=0.1, test_size=0.1, method='random', sort_by_col = None, random_state=None):
    '''
    For a given input dataframe this prepares X_train, y_train, X_valid, y_valid, X_test, y_test for final model development

    Parameters:
    -----------
    df: 'dataframe', input dataframe
    target: 'str' , target variable
    train_size: 'float', proportion of train dataset
    valid_size: 'float', proportion of valid dataset
    test_size: 'float', proportion of test dataset
    method: 'str', default 'random'. 
    2 methods available ['random', 'sorted']. in sorted dataframe is sorted by the input column and then splitting is done
    sort_by_col : 'str', defaul None. Required when method = 'sorted'
    random_state : random_state for train_test_split


    Output:
    -------
    X_train, y_train, X_valid, y_valid, X_test, y_test

    '''
    total = train_size + valid_size + test_size
    if total>1:
        raise Exception(" Total of train_size + valid_size + test_size should be 1")
    else:
        
        if method=='random':
            df_train, df_rem = train_test_split(df, train_size=train_size, random_state=random_state)
            test_prop = test_size/(test_size+valid_size)
            df_valid, df_test = train_test_split(df_rem, test_size=test_prop, random_state=random_state)

            X_train, y_train = df_train.drop(columns=target).copy(), df_train[target].copy()
            X_valid, y_valid = df_valid.drop(columns=target).copy(), df_valid[target].copy()
            X_test, y_test = df_test.drop(columns=target).copy(), df_test[target].copy()
        
        if method == 'sorted':
            train_index = int(len(df)*train_size)
            valid_index = int(len(df)*valid_size)
            
            df.sort_values(by = sort_by_col, ascending=True, inplace=True)
            df_train = df[0:train_index]
            df_rem = df[train_index:]
            df_valid = df[train_index:train_index+valid_index]
            df_test = df[train_index+valid_index:]

            X_train, y_train = df_train.drop(columns=target).copy(), df_train[target].copy()
            X_valid, y_valid = df_valid.drop(columns=target).copy(), df_valid[target].copy()
            X_test, y_test = df_test.drop(columns=target).copy(), df_test[target].copy()
            
            
        return X_train, y_train, X_valid, y_valid, X_test, y_test


def all_classifiers(X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, threshold_by = 'ROC AUC' ,verbose = True):
    '''
    Runs all the models

    Parameters:
    -----------
    threshold_by : 'str', default = 'ROC AUC Score'. Choose values from
    ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']

    multi_processing : bool, default True
    method : str, default concurrent
        'pool' : use pool from multiprocessing
        'concurrent' : use concurrent from features

    '''
    threshold_by = threshold_by + ' Score'

    clf_models = [
        # Ensemble models
        (1, 'AdaBoostClassifier', sklearn.ensemble.AdaBoostClassifier),
        (2, 'BaggingClassifier', sklearn.ensemble.BaggingClassifier),
        (3, 'ExtraTreesClassifier', sklearn.ensemble.ExtraTreesClassifier),
        (4, 'GradientBoostingClassifier', sklearn.ensemble.GradientBoostingClassifier),
        #('HistGradientBoostingClassifier', sklearn.ensemble.HistGradientBoostingClassifier),
        (5, 'HistGradientBoostingClassifier', sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier),
        (6, 'RandomForestClassifier', sklearn.ensemble.RandomForestClassifier),

        # Linear Classifiers
        (7, 'LogisticRegression', sklearn.linear_model.LogisticRegression),
        (8, 'LogisticRegressionCV', sklearn.linear_model.LogisticRegressionCV),
        (9, 'Perceptron', sklearn.linear_model.Perceptron), # no predict_proba
        (10, 'RidgeClassifier', sklearn.linear_model.RidgeClassifier), # no predict proba
        (11, 'RidgeClassifierCV', sklearn.linear_model.RidgeClassifierCV), # no predict proba

        # Naive Bayes
        (12, 'BernoulliNB', sklearn.naive_bayes.BernoulliNB),
        (13, 'GaussianNB', sklearn.naive_bayes.GaussianNB),
        (14, 'ComplementNB', sklearn.naive_bayes.ComplementNB),
        (15, 'MultinomialNB', sklearn.naive_bayes.MultinomialNB),

        # Nearest Neighbors
        (16, 'KNeighborsClassifier', sklearn.neighbors.KNeighborsClassifier),
        (17, 'NearestCentroid', sklearn.neighbors.NearestCentroid), # no predict proba
        (18, 'RadiusNeighborsClassifier', sklearn.neighbors.RadiusNeighborsClassifier),

        # Neural Network
        (19, 'MLPClassifier', sklearn.neural_network.MLPClassifier),

        # Support Vector Machines (SVM)
        (20, 'LinearSVC', sklearn.svm.LinearSVC), # no predict proba
#        (21, 'NuSVC', sklearn.svm.NuSVC), # no predict proba
        (22, 'SVC', sklearn.svm.SVC), # no predict proba

        # Decision Trees
        (23, 'DecisionTreeClassifier', sklearn.tree.DecisionTreeClassifier),
        (24, 'ExtraTreeClassifier', sklearn.tree.ExtraTreeClassifier)]

    model_grp1 = ['AdaBoostClassifier', 'BaggingClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier',
                  'HistGradientBoostingClassifier', 'RandomForestClassifier', 'LogisticRegression', 'LogisticRegressionCV',
                  'BernoulliNB', 'GaussianNB', 'ComplementNB', 'MultinomialNB', 'KNeighborsClassifier', 'MLPClassifier',
                  'DecisionTreeClassifier', 'ExtraTreeClassifier']

    model_grp2 = ['Perceptron', 'RidgeClassifier', 'RidgeClassifierCV', 'NearestCentroid', 
                  'LinearSVC', 'NuSVC', 'SVC']

    model_results = []
    for clf in clf_models:

        (sr, model_name, alg) = clf

        if model_name in model_grp1:
            
            start = time.perf_counter()

            #Instantiate
            model = alg()
            #fit
            model.fit(X_train, y_train)
            #Predict
            y_train_prob = model.predict_proba(X_train)[:,1]
            y_valid_prob = model.predict_proba(X_valid)[:,1]
            y_test_prob = model.predict_proba(X_test)[:,1]
            #Calculate threshold
            valid_result_df = threshold_evaluation(y_true=y_valid, y_prob=y_valid_prob, start=0, end=1, step_size=.05)
            id_max = valid_result_df[threshold_by].idxmax()
            threshold = valid_result_df.loc[id_max, 'Threshold']

            end = time.perf_counter()
            run_time= round(end-start, 2)

            train_res = metrics_evaluation(y_true=y_train, y_pred_prob=y_train_prob, threshold=threshold, df_type='train')
            train_res['Algorithm'] = model_name
            train_res['Run Time'] = run_time
            model_results.append(train_res)

            valid_res = metrics_evaluation(y_true=y_valid, y_pred_prob=y_valid_prob, threshold=threshold, df_type='valid')
            valid_res['Algorithm'] = model_name
            valid_res['Run Time'] = run_time
            model_results.append(valid_res)

            test_res= metrics_evaluation(y_true=y_test, y_pred_prob=y_test_prob, threshold=threshold, df_type='test')
            test_res['Algorithm'] = model_name
            test_res['Run Time'] = run_time
            model_results.append(test_res)

        if model_name in model_grp2:
            
            start = time.perf_counter()

            #Instantiate
            model = alg()
            #fit
            model.fit(X_train, y_train)
            #Predict
            y_train_pred = model.predict(X_train)
            y_valid_pred = model.predict(X_valid)
            y_test_pred = model.predict(X_test)
            #Calculate threshold -- here it is always 0.5
            threshold = 0.5

            end = time.perf_counter()
            run_time= round(end-start, 2)

            train_res = metrics_evaluation(y_true=y_train, y_pred=y_train_pred, threshold=threshold, df_type='train')
            train_res['Algorithm'] = model_name
            train_res['Run Time'] = run_time
            model_results.append(train_res)

            valid_res = metrics_evaluation(y_true=y_valid, y_pred=y_valid_pred, threshold=threshold, df_type='valid')
            valid_res['Algorithm'] = model_name
            valid_res['Run Time'] = run_time
            model_results.append(valid_res)

            test_res= metrics_evaluation(y_true=y_test, y_pred=y_test_pred, threshold=threshold, df_type='test')
            test_res['Algorithm'] = model_name
            test_res['Run Time'] = run_time
            model_results.append(test_res)

        
        if verbose:
            print(model_name , 'executed | Run time is', run_time, 'secs' )


    #print(model_results)
    model_results_df = pd.DataFrame(model_results)
    model_results_df = model_results_df[['Algorithm', 'Run Time', 'Dataset', 'No obs', 'Threshold',
                                         'TP', 'FP', 'TN', 'FN' , 'Accuracy Score', 'Precision Score', 
                                         'Recall Score', 'F1 Score', 'ROC AUC Score']]
    return model_results_df

def execute_model(model, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test, eval_metrics='ROC AUC Score'):
    model_results = []
    start = time.perf_counter()
    #Training
    model.fit(X_train, y_train)
    
    #Predict
    y_train_prob = model.predict_proba(X_train)[:,1]
    y_valid_prob = model.predict_proba(X_valid)[:,1]
    y_test_prob = model.predict_proba(X_test)[:,1]
    
    #Calculate threshold
    valid_result_df = threshold_evaluation(y_true=y_valid, y_prob=y_valid_prob, start=0, end=1, step_size=.05)
    id_max = valid_result_df[eval_metrics].idxmax()
    threshold = valid_result_df.loc[id_max, 'Threshold']

    end = time.perf_counter()
    run_time= round(end-start, 2)

    train_res = metrics_evaluation(y_true=y_train, y_pred_prob=y_train_prob, threshold=threshold, df_type='train')
    train_res['Algorithm'] = model_name
    train_res['Run Time'] = run_time
    model_results.append(train_res)

    valid_res = metrics_evaluation(y_true=y_valid, y_pred_prob=y_valid_prob, threshold=threshold, df_type='valid')
    valid_res['Algorithm'] = model_name
    valid_res['Run Time'] = run_time
    model_results.append(valid_res)

    test_res= metrics_evaluation(y_true=y_test, y_pred_prob=y_test_prob, threshold=threshold, df_type='test')
    test_res['Algorithm'] = model_name
    test_res['Run Time'] = run_time
    model_results.append(test_res)

    #print(model_results)
    model_results_df = pd.DataFrame(model_results)
    model_results_df = model_results_df[['Algorithm', 'Run Time', 'Dataset', 'No obs', 'Threshold',
                                     'TP', 'FP', 'TN', 'FN' , 'Accuracy Score', 'Precision Score', 
                                     'Recall Score', 'F1 Score', 'ROC AUC Score']]
    return model_results_df


'''
def all_classifiers(X_train, y_train, X_test=None, y_test=None, multi_processing=True, mp_method = 'concurrent', parameter_tuning=False, threshold=0.5, verbose=False):

    clf_models = [
    # Ensemble models
    (1, 'AdaBoostClassifier', sklearn.ensemble.AdaBoostClassifier),
    (2, 'BaggingClassifier', sklearn.ensemble.BaggingClassifier),
    (3, 'ExtraTreesClassifier', sklearn.ensemble.ExtraTreesClassifier),
    (4, 'GradientBoostingClassifier', sklearn.ensemble.GradientBoostingClassifier),
    #('HistGradientBoostingClassifier', sklearn.ensemble.HistGradientBoostingClassifier),
    (5, 'HistGradientBoostingClassifier', sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier),
    (6, 'RandomForestClassifier', sklearn.ensemble.RandomForestClassifier),

    # Linear Classifiers
    (7, 'LogisticRegression', sklearn.linear_model.LogisticRegression),
    (8, 'LogisticRegressionCV', sklearn.linear_model.LogisticRegressionCV),
    (9, 'Perceptron', sklearn.linear_model.Perceptron),
    (10, 'RidgeClassifier', sklearn.linear_model.RidgeClassifier),
    (11, 'RidgeClassifierCV', sklearn.linear_model.RidgeClassifierCV),

    # Naive Bayes
    (12, 'BernoulliNB', sklearn.naive_bayes.BernoulliNB),
    (13, 'GaussianNB', sklearn.naive_bayes.GaussianNB),
    (14, 'ComplementNB', sklearn.naive_bayes.ComplementNB),
    (15, 'MultinomialNB', sklearn.naive_bayes.MultinomialNB),

    # Nearest Neighbors
    (16, 'KNeighborsClassifier', sklearn.neighbors.KNeighborsClassifier),
    (17, 'NearestCentroid', sklearn.neighbors.NearestCentroid),
    (18, 'RadiusNeighborsClassifier', sklearn.neighbors.RadiusNeighborsClassifier),

    # Neural Network
    (19, 'MLPClassifier', sklearn.neural_network.MLPClassifier),

    # Support Vector Machines (SVM)
    (20, 'LinearSVC', sklearn.svm.LinearSVC),
    (21, 'NuSVC', sklearn.svm.NuSVC),
    (22, 'SVC', sklearn.svm.SVC),

    # Decision Trees
    (23, 'DecisionTreeClassifier', sklearn.tree.DecisionTreeClassifier),
    (24, 'ExtraTreeClassifier', sklearn.tree.ExtraTreeClassifier)]

    #More Advanced Models
    try:
        import lightgbm
        clf_models.append((25, 'LightGBM', lightgbm.LightGBMClassifier))
    except:
        lightgbm = None

    try:
        import xgboost
        clf_models.append((26, 'XGBoost', xgboost.XGBoostClassifier))
    except:
        xgboost = None


    if xgboost is None:
            raise ImportError('No module named xgboost')

    if lightgbm is None:
            raise ImportError('No module named lightgbm')

    

    def execute_model(model):    

        (sr, name, alg) = model
        try:
            start = time.perf_counter()
            clf=alg()
            clf.fit(X_train, y_train)
            y_test_predict = clf.predict(X_test)
            end = time.perf_counter()
            run_time= round(end-start, 2)
            # Calculating the evaluation metrics
            accuracy_score = accuracy_score(y_test, y_test_predict)
            auc_score=roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            tn = confusion_matrix(y_test, y_test_predict)[0][0]
            fp = confusion_matrix(y_test, y_test_predict)[0][1]
            fn = confusion_matrix(y_test, y_test_predict)[1][0]
            tp = confusion_matrix(y_test, y_test_predict)[1][1]
            
            TPR = tp / (tp + fn)
            FPR = fp / (fp + tn)
            FNR = fn / (fn + tp)
            TNR = tn / (tn + fp) 
            if tp == fp == 0:
                PPV, FDR = 0, 0
            else:
                PPV = tp / (tp + fp)
                FDR = fp / (tp + fp)
            if tn == fn == 0:
                NPV, FOR = 0, 0
            else:
                NPV = tn / (tn + fn)
                FOR = fn / (tn + fn)
            if PPV == TPR == 0: f1 = 0
            else: f1 = 2 * (PPV * TPR) / (PPV + TPR)
                
            if verbose:
                print(sr, ". ", name, " - executed | Time Taken :", run_time)
            result = [name, 1, run_time, accuracy_score, auc_score, tp, tn, fp, fn, PPV, TPR, f1, NPV, FPR, TNR, FNR, FDR, FOR]

            return result
        
        except:
            if verbose:
                print(sr, ". ", name, " - not executed")
            result = [name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
        return result


    train_results=[]
    test_results=[]

    if multi_processing == False:
        for model in clf_models:
            test_results.append(execute_model(model))

    if multi_processing == True and mp_method == 'concurrent':
        with futures.ProcessPoolExecutor() as executor:
            test_results = list(tqdm.tqdm(executor.map(execute_model, clf_models), total = len(clf_models), desc = 'Running Models')) 

    if multi_processing == True and mp_method == 'pool':
        pool = Pool()
        test_results = list(tqdm.tqdm(pool.imap(execute_model, clf_models), total = len(clf_models), desc = 'Running Models')) 


    test_result_df= pd.DataFrame(test_results)
    test_result_df.columns=["ALGORITHM","EXECUTED", "RUN_TIME", "ACCURACY_SCORE", "ROC_AUC_SCORE", "TP", "TN", "FP", "FN","PRECISION", "RECALL", "F1_SCORE","NPV","FALL_OUT","SPECIFICITY","MISS_RATE","FDR","FOR"]
    
    return test_result_df
'''
