# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:32:13 2022

@author: mn46
"""


import sys
import os
import re
import pandas as pd

def notes_fnc(df, notes, col, path):
    
    sys.path.insert(0, path) # insert path
    
    def merge_notes(n, col):
         
        n[col] = n[col].astype(str).apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))
        
        n[col][n[col].astype(str) == 'nan'] = ' ' 
        
        # group all notes
      
        n.Date = n.Date.astype('datetime64[ns]')
        
        # ------------------------------------------------------
        # Group notes per MRN by ContactDateRealNBR
        # ------------------------------------------------------
        
        n = n.sort_values(['Date',"NoteID","LineNBR"], ascending=[True,True,True])
        
        n[col] = n[col].astype(str)
        
        #added
        n[col] = n[col].str.rjust(5)
        n[col] = n[col].str.ljust(5)
        
        n = n.groupby(['PatientID','Author','Date','NoteID'])[col].sum().reset_index()
        
        #added
        n[col] = n[col].apply(lambda x: " ".join(x.split())) # removes duplicated spaces
        
        n = n.sort_values(['Date',"NoteID"], ascending=[True,True])
        
        #added
        n[col] = n[col].str.rjust(5)
        n[col] = n[col].str.ljust(5)
        
        n = n.groupby(['PatientID','Author','Date'])[col].sum().reset_index()
        
        #added
        n[col] = n[col].apply(lambda x: " ".join(x.split())) # removes duplicated spaces
        
        n = n.sort_values(['Date'], ascending=[True])
        
        #added
        n[col] = n[col].str.rjust(5)
        n[col] = n[col].str.ljust(5)
        
        n = n.groupby(['PatientID','Date'])[col].sum().reset_index()
        
        return n
    
    notes = merge_notes(notes, col)
     
    #Join notes to demographics, icds and meds
    
    df = pd.merge(df,notes, on=['PatientID', 'Date'])
    
          
    # Build text features matrix
    
    from build_binary_features import build_matrix_features
    
    df2 = build_matrix_features(df, col)
    
    cs = list(df2.columns)
    
    c = list(df.columns)
    
    for i in c:
        cs.remove(i)
        
    for i in cs:
        df2[i][df2[i].astype(str) == 'nan'] = 0
        df2[i][df2[i].astype(int) >= 1] = 1
 
     #Save this dataset
    df2.to_csv(os.path.join(path,'matrix_build.csv'))
    
    return df2
