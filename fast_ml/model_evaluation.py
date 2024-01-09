import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time


def model_save (model, model_name):
    file_name = model_name+'.joblib'
    dump(model, file_name)
    
def model_load (model_name):
    file_name = model_name+'.joblib'

    return load(file_name)

def get_fi(model, X):
    fi = pd.DataFrame(model.feature_importances_, X.columns, columns=['feature_imp'])
    fi.sort_values(by ='feature_imp', ascending=False, inplace=True)
    fi['cum_fi'] = fi['feature_imp'].cumsum()
    fi['cum_perc'] = fi['cum_fi']/fi['feature_imp'].sum()
    return fi


def plot_confidence_interval_for_data (model, X):
    """
    Pass 10 - 15 datapoints for better visualization. 
    This function plots the confidence interval of predictive value for the provided datapoints

    Parameters:
    -----------
        model : model that is built
        X : datapoints for evaluation

    Returns:
    --------
        Plot

    """
    preds = np.stack([t.predict(X) for t in model.estimators_], axis=1)
    preds_ds = pd.DataFrame()
    preds_ds['mean'] = preds.mean(axis=1)
    preds_ds['std'] = preds.std(axis=1)

    fig = plt.figure(figsize=(15,6))
    my_xticks = ['datapoint ' + str(i+1) for i in list(preds_ds.index)]
    plt.errorbar(x = preds_ds.index, y=preds_ds['mean'], yerr=preds_ds['std'], 
             fmt='o', color='blue', ecolor='lightblue', capsize=3)
    plt.title('Confidence Interval for the predicted value')
    plt.xticks(preds_ds.index, my_xticks)
    for i in list(preds_ds.index):
        m, std = round(preds_ds['mean'][i],1), round(preds_ds['std'][i],2)
        s=f' pred={m} \n std dev= {std}'
        plt.text(x = i, y=preds_ds['mean'][i], s=s ) 
    plt.show()


def plot_confidence_interval_for_variable (model, X, y, variable):
    """
    This function plots the confidence interval of predictive value for the provided variable

    Parameters:
    -----------
        model : model that is built
        X : datapoints 
        y : actual value
        variable : variable for evaluation

    Returns:
    --------
        Plot

    """

    preds = np.stack([t.predict(X) for t in model.estimators_], axis=1)
    X_ds_new = X.copy()
    X_ds_new['actual'] = y
    X_ds_new['pred'] = np.mean(preds, axis=1)
    X_ds_new['pred_std'] = np.std(preds, axis=1)

    X_ds_grp = X_ds_new.groupby(variable)['actual', 'pred', 'pred_std'].agg('mean')
    X_ds_grp['count'] = X_ds_new[variable].value_counts()

    print (f'Average Predicted value and Std Dev by : {variable}')
    display(X_ds_grp)
    print ('')
    print (f'Distribution of Predicted value by : {variable}')
    sns.catplot(x=variable, y='pred', data=X_ds_new, kind='box')
    plt.show()


def threshold_evaluation(y_true, y_prob, start=0, end=1, step_size=0.1):
    """
    This function produces various model evaluation metrics at various values of threshold. 
    The values of threshold are customizable using parameters 'start', 'end', 'nsteps'

    Parameters:
    -----------
        y_true : 'array', actual value of y (this could be y_train, y_valid, or y_test)
        y_prob : 'array', predicted value of y (this could be from train, valid or test)
        start : 'int', default = 0. starting point for threshold values
        end : 'int', default = 1. End point for threshold values
        step_size : 'float', default = 0.1 | Step size for incrementing the threshold values

    Returns:
    --------
        df : 'dataframe', dataframe with various model evaluation metrics 
    """
    threshold_list = np.arange(start,end,step_size)
    result = []
    
    for t in threshold_list:
        y_pred = (y_prob>=t).astype(int)
        tn = metrics.confusion_matrix(y_true, y_pred)[0][0]
        fp = metrics.confusion_matrix(y_true, y_pred)[0][1]
        fn = metrics.confusion_matrix(y_true, y_pred)[1][0]
        tp = metrics.confusion_matrix(y_true, y_pred)[1][1]

        accuracy_scr = metrics.accuracy_score(y_true, y_pred)
        precision_scr = metrics.precision_score(y_true, y_pred)
        recall_scr = metrics.recall_score(y_true, y_pred)
        f1_scr = metrics.f1_score(y_true, y_pred)
        roc_auc_scr = metrics.roc_auc_score(y_true, y_pred)

        result.append((t, tp, fp, tn, fn, accuracy_scr, precision_scr, recall_scr, f1_scr, roc_auc_scr))
    
    result_df = pd.DataFrame(result)
    result_df.columns = ['Threshold','TP', 'FP', 'TN', 'FN' , 'Accuracy Score', 'Precision Score', 
                         'Recall Score', 'F1 Score', 'ROC AUC Score']
    return result_df

def metrics_evaluation(y_true, y_pred_prob=None, y_pred=None, threshold=None, df_type='train'):
    """
    This function produces various model evaluation metrics at various values of threshold. 
    The values of threshold are customizable using parameters 'start', 'end', 'nsteps'

    Parameters:
    -----------
        y_true : 'array', actual value of y (this could be y_train, y_valid, or y_test)
        y_pred_prob : 'array', predicted probability of y (this could be from train, valid or test)
        y_pred : 'array', predicted value of y (this could be from train, valid or test)
        threshold : 'float', threshold value at which predicted probability needs to be converted to predictions
        df_type : 'str', Usual values are 'train', 'valid, 'test'

    Returns:
    --------
        result : 'list', list with various model evaluation metrics 
    """
    input_y_pred = y_pred 

    if input_y_pred is None:
        y_pred = (y_pred_prob>=threshold).astype(int)
    
    tn = metrics.confusion_matrix(y_true, y_pred)[0][0]
    fp = metrics.confusion_matrix(y_true, y_pred)[0][1]
    fn = metrics.confusion_matrix(y_true, y_pred)[1][0]
    tp = metrics.confusion_matrix(y_true, y_pred)[1][1]

    accuracy_scr = metrics.accuracy_score(y_true, y_pred)
    precision_scr = metrics.precision_score(y_true, y_pred)
    recall_scr = metrics.recall_score(y_true, y_pred)
    f1_scr = metrics.f1_score(y_true, y_pred)
    
    if input_y_pred is None:
        roc_auc_scr = metrics.roc_auc_score(y_true, y_pred)
    else:
        roc_auc_scr = np.nan

    result = {'Dataset': df_type, 'No obs': len(y_true), 'Threshold': threshold,
              'TP':tp, 'FP': fp, 'TN': tn, 'FN':fn , 
              'Accuracy Score':accuracy_scr, 'Precision Score':precision_scr, 
              'Recall Score':recall_scr, 'F1 Score':f1_scr, 'ROC AUC Score':roc_auc_scr}

    return result


