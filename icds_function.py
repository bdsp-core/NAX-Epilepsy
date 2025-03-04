# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:25:43 2022

@author: mn46
"""

import sys
import os
import re
import numpy as np  
import pandas as pd

def icds_fnc(icds, d, g, path):
    
    sys.path.insert(0, path) # insert path
    
    icds = icds[icds.PatientID.isin(d.PatientID)].reset_index().drop(columns='index')
      
    g = g[['DiagnosisID','grouping ICD ']].drop_duplicates()
    
    # g['grouping ICD '].drop_duplicates()
    g['grouping ICD '] = g['grouping ICD '].str.lower()
    g['grouping ICD '][g['grouping ICD '].astype(str).str.contains('sequelae')] = 'seizures sequelae'
    g['grouping ICD '] = g['grouping ICD '].astype(str).str.replace('_',' ')
    
    g['group'] = 0
    g['group'][g['grouping ICD '].astype(str) == 'syncope'] = 1
    g['group'][g['grouping ICD '].astype(str) == 'seizures sequelae'] = 2
    g['group'][g['grouping ICD '].astype(str) == 'pregnancy and seizure risk'] = 3
    g['group'][g['grouping ICD '].astype(str) == 'nonepileptic episode'] = 4
    g['group'][g['grouping ICD '].astype(str) == 'epilepsy and recurrent seizures'] = 5
    g['group'][g['grouping ICD '].astype(str) == 'convulsions seizures'] = 6
    
    g = g[g.DiagnosisID.isin(icds.DiagnosisID)]
    
    icds = pd.merge(icds,g, on='DiagnosisID', how='inner')
    
    # create cols for each icd
    data = icds['grouping ICD ']
    a = pd.get_dummies(data, prefix=None, prefix_sep='_')
    
    icds = pd.concat([icds, a], axis = 1)
    
    icds = icds.drop(columns=['DiagnosisID','grouping ICD ', 'group']).drop_duplicates()
    
    icds = icds.groupby(['PatientID', 'Date'])[a.columns].sum().reset_index()
    
    icds=icds[['PatientID', 'Date', 'convulsions seizures',
            'epilepsy and recurrent seizures','nonepileptic episode', 'syncope']].drop_duplicates().reset_index().drop(columns='index')

    
    # Allow 1 day before and 1 day after outpatient visit for ICD assignment    
     
    df = pd.merge(d,icds, on=['PatientID'], how='outer')
 
    df['diff'] = df.Date_x.astype('datetime64[ns]') - df.Date_y.astype('datetime64[ns]')
    
    df['diff'] = df['diff'].astype(str).str.extract('(-?\d+) days')
    
    
    df = df[((df['diff'].astype(float)  == -1) | (df['diff'].astype(float)  == 0) | (df['diff'].astype(float)  == 1))]

    df = df.rename(columns={'Date_x':'Date'})
    
    
    df_aux = df.groupby(['PatientID', 'Date'])[['convulsions seizures', 'epilepsy and recurrent seizures',
       'nonepileptic episode', 'syncope']].sum().reset_index()
                                                                                  
    df = pd.merge(d,df_aux,on=['PatientID', 'Date'], how='outer')

                                        
    cols = ['convulsions seizures',
            'epilepsy and recurrent seizures','nonepileptic episode', 'syncope']
    
    for col in cols:
        df[col][df[col].astype(str) == 'nan'] = 0
    
    df['convulsions seizures'][df['nonepileptic episode'] >= 1] = 1
    
    cols = ['convulsions seizures',
            'epilepsy and recurrent seizures', 'syncope']
    
    for col in cols:
        df[col][df[col]>= 1] = 1
        
    df = df.drop(columns=['nonepileptic episode'])
                                                                                      
    df['n_icds'] =  np.sum(df[['convulsions seizures', 'epilepsy and recurrent seizures', 'syncope']], axis=1)
  
    df.to_csv(os.path.join(path,'dataset_icds_encounter.csv'), index=False,sep=',') 
    
    return df
