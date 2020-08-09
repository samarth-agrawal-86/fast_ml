import pandas as pd
import numpy as np
from joblib import dump, load


def model_save (model, model_name):
    file_name = model_name+'.joblib'
    dump(model, file_name)
    
def model_load (model_name):
    file_name = model_name+'.joblib'

    return load(file_name)


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



