import sys
import os
import numpy as np  
import pandas as pd

def icds_fnc(icds, probs, d, g, path):
    
    sys.path.insert(0, path) # insert path
    
   
    icds.ContactDTS = icds.ContactDTS.astype("datetime64[ns]")

    probs = probs[probs.ProblemStatusDSC != 'Deleted']

    probs = probs.rename(columns={'DiagnosisDTS':"ContactDTS"})
    probs.ContactDTS = probs.ContactDTS.astype("datetime64[ns]")

    cs = ['PatientID', 'ContactDTS', 'ICD10', 'ICD9']

    cols = ['ICD10','ICD9']

    for i in cols:
        icds[i] = icds[i].astype(str)
        probs[i] = probs[i].astype(str)

    icds = pd.merge(icds[cs], probs[cs], on=cs, how='outer').drop_duplicates()


    cols = ['ICD10','ICD9']

    for i in cols:
        g[i] = g[i].astype(str)
       
    icds = pd.merge(icds,g, on=['ICD10','ICD9'], how='inner')

    # create cols for each icd
    data = icds['groups']
    a = pd.get_dummies(data, prefix=None, prefix_sep='_')
    
    icds = pd.concat([icds, a], axis = 1)
    
    icds = icds.drop(columns=['ICD10','ICD9', 'groups']).drop_duplicates()
    
    icds = icds.groupby(['PatientID', 'ContactDTS'])['epilepsy and recurrent seizures',
           'convulsions seizures'].sum().reset_index()
    
    cols = ['epilepsy and recurrent seizures', 'convulsions seizures']
    
    for col in cols:
        icds[col][icds[col]> 1] = 1 
  
    icds = icds.rename(columns={'ContactDTS':'Date'})
    
    # Allow 1 day before and 1 day after current visit
     
    df = pd.merge(d[['PatientID','ContactDTS']].drop_duplicates(),icds, on=['PatientID'], how='outer')
 
    df['diff'] = df.ContactDTS.astype("datetime64[ns]") - df.Date.astype("datetime64[ns]")
    
    df['diff'] = df['diff'].astype(str).str.extract(r'(-?\d+) days')
    
      
    df = df[((df['diff'].astype(float)  == -1) | (df['diff'].astype(float)  == 0) | (df['diff'].astype(float)  == 1))]

    
    df_aux = df.groupby(['PatientID', 'ContactDTS'])['convulsions seizures', 'epilepsy and recurrent seizures'].sum().reset_index()
        
        
    df = pd.merge(d[['PatientID','ContactDTS']].drop_duplicates(),df_aux,on=['PatientID', 'ContactDTS'], how='outer')

                                        
    cols = list(df.drop(columns=['PatientID', 'ContactDTS']).columns)
    
    for col in cols:
        df[col][df[col].astype(str) == 'nan'] = 0
    
   
    cols = list(df.drop(columns=['PatientID', 'ContactDTS']).columns)
    
    for col in cols:
        df[col][df[col]>= 1] = 1
        
    
    df['n_icds'] =  np.sum(df[cols], axis=1)


    df.to_csv(os.path.join(path,'dataset_icds.csv'), index=False,sep=',') 
    
    return df
