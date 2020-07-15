# Set the working directory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Main function which will be called :
## 
## Inputs: 1. df - dataframe to be analysed
##         2. iqrLevel - Level to cal bench marks = 1.5 default    

## Retuns: Dataframe with summary of outliers present in the original dataframe

def Outlier_Analysis(df,iqrLevel, IsGaussian):  #Inputs are dataframe and level to cal bench marks = 1.5 default
    # Step 1: Split numeric data into separate df
    num_data = df.select_dtypes(include=[np.number])
    dict_UBLB ={}
    dict_LogUbLb ={}
    dict_TotUBLB = {}
        # Creating dictionary to avoid multiple calls to Method CalUbLB
    dict_UBLB= CalUbLB(num_data,iqrLevel,0,IsGaussian)
    dict_LogUbLb = CalUbLB(num_data,iqrLevel,1,IsGaussian)
    dict_TotUBLB = {"Total_UBLB":np.add(dict_UBLB['Count_UB'],dict_UBLB['Count_LB']),
                    "Total_LogUBLB":np.add(dict_LogUbLb['Count_UB'],dict_LogUbLb['Count_LB'])}   


    DataFrameDict ={'Variable_Name':num_data.columns,
                    'Count_UB_outliers':dict_UBLB['Count_UB'],
                    'Count_LB_outliers':dict_UBLB['Count_LB'],
                    'Count_LogUB_outliers':dict_LogUbLb['Count_UB'],
                    'Count_LogLB_outliers':dict_LogUbLb['Count_LB'],                
                    'Total_Outliers':dict_TotUBLB['Total_UBLB'],
                    'Total_log_Outliers':dict_TotUBLB['Total_LogUBLB']}
    
    return pd.DataFrame(data = DataFrameDict,columns=DataFrameDict.keys())
    
    
    
    
    ## Inputs: 1. df - dataframe to be analysed
##         2. iqrLevel - Level to cal bench marks = 1.5 default
##         3. iflogmethod - 1:Flag for Log transformation 0: otherwise

## Returns: dictionary for count of UB and LB.
def CalUbLB(num_data,iqrLevel,iflogmethod, isGauss):
    dict_return ={}
    Count_UB=[]
    Count_LB=[]
    
    if iflogmethod==1:
        for i in range(len(num_data.columns)):            
            num_data[num_data.columns[i]] = np.where(num_data[num_data.columns[i]] == 0,1,num_data[num_data.columns[i]])
        num_data = np.log(num_data) 
        
    for i in range(len(num_data.columns)): 
        if isGauss == 1:         
            UB = num_data[num_data.columns[i]].mean() + (3 * num_data[num_data.columns[i]].std())
            LB = num_data[num_data.columns[i]].mean() - (3 * num_data[num_data.columns[i]].std())
   
        else:         
            Q1= num_data[num_data.columns[i]].quantile(0.25)
            Q3 = num_data[num_data.columns[i]].quantile(0.75)
            IQR = Q3 - Q1        
            UB = min(max(num_data[num_data.columns[i]]),Q3 + iqrLevel*IQR)           
            LB = max(min(num_data[num_data.columns[i]]),Q1 - iqrLevel* IQR)
        Count_UB.append(num_data[num_data.columns[i]][num_data[num_data.columns[i]] > UB].count() ) 
        Count_LB.append(num_data[num_data.columns[i]][num_data[num_data.columns[i]] < LB].count())
    dict_return = {'Count_UB':Count_UB, 'Count_LB':Count_LB}
    return dict_return
          
        
