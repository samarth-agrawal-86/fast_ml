import pandas as pd
import numpy as np


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