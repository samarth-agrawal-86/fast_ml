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


def all_classifiers(X_train, y_train, X_valid, y_valid, X_test, y_test, verbose = True):
    '''
    Runs all the models

    Parameters:
    -----------
    multi_processing : bool, default True
    method : str, default concurrent
        'pool' : use pool from multiprocessing
        'concurrent' : use concurrent from features

    '''

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
    #    (9, 'Perceptron', sklearn.linear_model.Perceptron), # no predict_proba
    #    (10, 'RidgeClassifier', sklearn.linear_model.RidgeClassifier), # no predict proba
    #    (11, 'RidgeClassifierCV', sklearn.linear_model.RidgeClassifierCV), # no predict proba

        # Naive Bayes
        (12, 'BernoulliNB', sklearn.naive_bayes.BernoulliNB),
        (13, 'GaussianNB', sklearn.naive_bayes.GaussianNB),
        (14, 'ComplementNB', sklearn.naive_bayes.ComplementNB),
        (15, 'MultinomialNB', sklearn.naive_bayes.MultinomialNB),

        # Nearest Neighbors
        (16, 'KNeighborsClassifier', sklearn.neighbors.KNeighborsClassifier),
    #    (17, 'NearestCentroid', sklearn.neighbors.NearestCentroid), # no predict proba
    #    (18, 'RadiusNeighborsClassifier', sklearn.neighbors.RadiusNeighborsClassifier),

        # Neural Network
        (19, 'MLPClassifier', sklearn.neural_network.MLPClassifier),

        # Support Vector Machines (SVM)
    #    (20, 'LinearSVC', sklearn.svm.LinearSVC), # no predict proba
    #    (21, 'NuSVC', sklearn.svm.NuSVC), # no predict proba
    #    (22, 'SVC', sklearn.svm.SVC), # no predict proba

        # Decision Trees
        (23, 'DecisionTreeClassifier', sklearn.tree.DecisionTreeClassifier),
        (24, 'ExtraTreeClassifier', sklearn.tree.ExtraTreeClassifier)]

    model_results = []
    for clf in clf_models:

        (sr, model_name, alg) = clf
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
        id_max = valid_result_df['ROC AUC Score'].idxmax()
        threshold = valid_result_df.loc[id_max, 'Threshold']

        end = time.perf_counter()
        run_time= round(end-start, 2)

        train_res = metrics_evaluation(y_true=y_train, y_prob=y_train_prob, threshold=threshold, df_type='train')
        train_res['Algorithm'] = model_name
        train_res['Run Time'] = run_time
        model_results.append(train_res)

        valid_res = metrics_evaluation(y_true=y_valid, y_prob=y_valid_prob, threshold=threshold, df_type='valid')
        valid_res['Algorithm'] = model_name
        valid_res['Run Time'] = run_time
        model_results.append(valid_res)

        test_res= metrics_evaluation(y_true=y_test, y_prob=y_test_prob, threshold=threshold, df_type='test')
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
