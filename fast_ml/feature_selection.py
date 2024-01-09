import pandas as pd
import numpy as np
from sklearn.metrics import *




def get_constant_features(df, threshold=0.99, dropna=False):
    '''
    For a given dataframe, identify the constant and quasi constant features.
    To get all the constant & quasi constant features in a list - constant_features_df['Var'].to_list()
    
    Parameters:
    -----------
        df: 'dataframe'
        threshold: 'float'. default = 0.99
        dropna: 'bool'. default = false
        
    Returns:
    --------
        constant_features_df: 'dataframe'
    '''
    constant_features = []
    constant_features_df = pd.DataFrame(columns=['Desc', 'Var', 'Value', 'Perc'])
    all_vars = list(df.columns)
    i=0
    for var in all_vars:
        s = df[var].value_counts(normalize=True, dropna=dropna)
        value = s.index[0]
        perc = s.iloc[0]
    
        if perc==1:
            constant_features_df.loc[i] = ['Constant', var, value, 100*perc]

        elif perc>threshold:
            constant_features_df.loc[i] = ['Quasi Constant', var, value, 100*perc]
    
        i=i+1
    
    constant_features_df = constant_features_df.sort_values(by='Perc', ascending=False, ignore_index=True) 

    return constant_features_df

def get_duplicate_features(df):
    '''
    For a given dataframe, identify the duplicate features
    To get all the constant & quasi constant features in a list - duplicate_features_df['feature2'].to_list()
    
    Parameters:
    -----------
        df: 'dataframe'
        
    Returns:
    --------
        duplicate_features_df: 'dataframe'
    
    '''
    duplicate_features_df = pd.DataFrame(columns = ['Desc', 'feature1', 'feature2'])
    duplicate_features_ = []
    ix=0
    for i,v1 in enumerate(df.columns,0):
        if v1 not in duplicate_features_:
            for v2 in df.columns[i+1:]:
                if df[v1].nunique() == df[v2].nunique():

                    # This check for duplicate values
                    if df[v1].equals(df[v2]):
                        duplicate_features_.append(v2)
                        duplicate_features_df.loc[ix] = ['Duplicate Values', v1, v2]
                        ix=ix+1
                    
                    # This check for duplicate index
                    elif df[[v1, v2]].drop_duplicates().shape[0] == df[v1].nunique():
                        duplicate_features_df.loc[ix] = ['Duplicate Index', v1, v2]
                        ix=ix+1
    duplicate_features_df = duplicate_features_df.sort_values(by='Desc', ascending=False, ignore_index=True)
    
    return duplicate_features_df

def get_duplicate_pairs (df):
    '''
    To get list of duplicate features from this dictionary run this command
    [item for sub_list in list(duplicate_pairs_.values()) for item in sub_list]
    
    '''
    duplicate_features_ = []
    duplicate_pairs_ = {}

    for i,v1 in enumerate(df.columns,0):
        duplicate_feat = []
        if v1 not in duplicate_features_:
            for v2 in df.columns[i+1:]:
                if df[v1].equals(df[v2]):
                    duplicate_features_.append(v2)
                    duplicate_feat.append(v2)
            if duplicate_feat:
                duplicate_pairs_[v1] = duplicate_feat
                
    return duplicate_pairs_

def get_correlated_pairs(df, threshold=0.9):
    
    df_corr = df.corr()
    df_corr = pd.DataFrame(df_corr.unstack())
    df_corr = df_corr.reset_index()
    df_corr.columns = ['feature1', 'feature2', 'corr']
    df_corr['abs_corr'] = df_corr['corr'].abs()

    #print('original corr dataframe Shape', df_corr.shape)

    # Removing correlation below the threshold
    df_corr = df_corr.query(f'abs_corr >= {threshold}')

    # Removing correlations within the same features
    df_corr = df_corr[~(df_corr['feature1']==df_corr['feature2'])]

    # Removing cases where first v1 was compared with v2 and then later v2 compared with v1
    for v1 in df_corr['feature1'].unique():
        for v2 in df_corr['feature2'].unique():
            drop_ix = df_corr[(df_corr['feature1']==v2) & (df_corr['feature2'] == v1)].index
            df_corr.drop(index=drop_ix, inplace=True)

    # Creating correlation groups        
    df_corr['corr_group'] = (df_corr.groupby(by='feature1').cumcount()==0).astype('int')
    df_corr['corr_group'] = df_corr['corr_group'].cumsum()

    # Formating changes
    df_corr.sort_values(by='corr_group', inplace=True)
    df_corr.reset_index(drop=True, inplace=True)
    df_corr = df_corr[[ 'corr_group', 'feature1', 'feature2', 'corr', 'abs_corr']]
    #print('Final corr dataframe Shape', df_corr.shape)
    
    return df_corr

def recursive_feature_elimination(model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    rfe_df = pd.DataFrame(columns = ['dropped_feature', 'num_features', 'train_roc', 'valid_roc', 'test_roc'])
    features_to_drop = []

    for i in range(0, len(X_train.columns)):
        X_train_c = X_train.copy()
        X_valid_c = X_valid.copy()
        X_test_c = X_test.copy()
        
        X_train_c = X_train_c.drop(columns = features_to_drop)
        X_valid_c = X_valid_c.drop(columns = features_to_drop)
        X_test_c = X_test_c.drop(columns = features_to_drop)

        #model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=SEED)
        model.fit(X_train_c, y_train)
        #print(model)

        #train
        y_train_pred = model.predict_proba(X_train_c)[:,1]
        train_roc = roc_auc_score(y_train, y_train_pred)
        #print('Train ROC Score:', train_roc)

        #valid
        y_valid_pred = model.predict_proba(X_valid_c)[:,1]
        valid_roc = roc_auc_score(y_valid, y_valid_pred)
        #print('Valid ROC Score:', valid_roc)
        
        #test
        y_test_pred = model.predict_proba(X_test_c)[:,1]
        test_roc = roc_auc_score(y_test, y_test_pred)
            
        data = {'feature': X_train_c.columns, 'fi': model.feature_importances_}
        fi = pd.DataFrame(data)
        fi.sort_values(by = 'fi', ascending=False, inplace=True)

        lowest_fi = list(fi['feature'])[-1]
        features_to_drop.append(lowest_fi)

        if i ==0:
            drop_f = 'None'
        else:
            drop_f = features_to_drop[-1]


        rfe_df.loc[i] = [drop_f, len(X_train_c.columns), train_roc, valid_roc, test_roc]

    print('Done')
    rfe_df['train_roc_rank'] =rfe_df['train_roc'].rank(method='min', ascending=False).astype('int')
    rfe_df['valid_roc_rank'] =rfe_df['valid_roc'].rank(method='min', ascending=False).astype('int')
    rfe_df['test_roc_rank'] =rfe_df['test_roc'].rank(method='min', ascending=False).astype('int')
        
    return rfe_df

def variables_clustering (df, variables, method):
    """
    This function helps in performing the variable clustering.
    
    'spearman': This evaluates the monotonic relationship between two continuous or ordinal variables.
    'pearson': This evaluates the linear relationship between two continuous variables.

    
    Parameter:
    ----------
    df : dataframe
        Dataframe for analysis
    variables : list type, optional
        List of all the variables for which clustering needs to be done. If not provided it will automatically select all the numerical analysis
    method : str, default 'spearman'
        'pearson' : For pearson correlation
        'spearman' : For spearman correlation
        
    Returns:
    --------
    Dendogram with hierarchial clustering for variables
    """

    cluster_df = df[variables]
    
    if method in ("Pearson", "pearson"):
        corr = cluster_df.corr(method='pearson')
        title = "Pearson Correlation"
        
    elif method in ("Spearman", "spearman"):
        corr = cluster_df.corr(method='spearman')
        #corr = spearmanr(cluster_df).correlation
        title = "Spearman Correlation"
        
    fig  = plt.figure(figsize=(16, int(len(variables)/2)))
    ax = fig.add_subplot(111)
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(corr_linkage, labels=variables, leaf_rotation=360, orientation ='left', ax = ax)
    dendro_idx = np.arange(0, len(dendro['ivl']))
    plt.title(title + ' - Hierarchial Clustering Dendrogram', fontsize = 17)
    ax.tick_params(axis='y', which='major', labelsize=10)
    plt.show()