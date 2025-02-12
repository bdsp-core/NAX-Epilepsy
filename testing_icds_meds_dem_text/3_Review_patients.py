# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:43:02 2023

@author: mn46
"""


import sys
import os
import re
import nltk
import numpy as np  
import pandas as pd

from dateutil.relativedelta import relativedelta


# Directory here
path = 'C:/Users/mn46/Downloads/Test_epilepsy_alg/testing/'
sys.path.insert(0, path) # insert path

df = pd.read_csv(os.path.join(path,'dataset_with_scores.csv'), index_col = 0)

df.Date = df.Date.astype("datetime64[ns]")


# groups

# First create n_meds_notes

aeds = ['acetazolamid', 'acth',
    'acthar', 'brivaracetam',
    'briviact', 'cannabidiol' , 'epidiolex',
    'carbamazepin', 'cbz', 'epitol', 'tegretol', 'equetro', 'teril',
     'carbatrol', 'tegretol', 'epitol', 'cenobam', 'xcopri',
     'clobazam', 'frisium', 'onfi', 'sympazan', 'clonazepam',
     'epitril', 'klonopin', 'rivotril', 'clorazep', 'tranxen',
     'xene', 'diazepam', 'valium' , 'diamox',
     'diastat', 'divalproex', 'depakot', 'eslicarbazepin', 'aptiom',
     'ethosuximid', 'zarontin', 'ethotoin', 'ezogabin', 'potiga',
     'felbam', 'felbatol', 'gabapentin', 'neurontin', 'gralis',
     'horiz', 'lacosamid', 'vimpat', 'lamotrigin', 'lamict',
     'levetiracetam', 'ltg', 'ige', 'tpm', 'oxc', 'lev', 'keppra', 'roweepra', 'spritam',
     'elepsia', 'lorazepam', 'ativan', 'methsuximid', 'methosuximid',
     'celontin', 'oxcarbazepin', 'trilept', 'oxtellar xr', 'perampanel',
    'fycompa', 'phenobarbit', 'luminol', 'lumin', 'phenytoin',
     'epanutin', 'dilantin', 'phenytek', 'pregabalin', 'lyrica',
     'primidon', 'mysolin', 'rufinamid', 'banzel', 'inovelon', 'percocet',
     'stiripentol', 'diacomit', 'tiagabin', 'gabitril', 'topiram', 'topamax',
     'topiram',  'qudexi', 'trokendi', 'valproat', 'valproic', 'wellbutrin',
     'convulex', 'depacon', 'depaken', 'orfiril', 'valpor', 'valprosid',
     'depakot', 'vigabatrin', 'sabril', 'vigadron', 'zonisamid', 'zonegran', 'xanax']

matrix = pd.DataFrame()
matrix[aeds] = 0 
matrix = matrix.add_suffix('_')

ls=[]
for i in list(matrix.columns):
    if i in list(df.columns):
        ls.append(i)

df['n_meds_notes'] = 0

df['n_meds_notes'] = np.sum(df[ls], axis=1) 

df['n_meds_total'] = 0

df['n_meds_total'] = df['n_meds'] + df['n_meds_notes'] 

 
df['icds_meds'] = 0
df['icds_no_meds'] = 0
df['no_icds_meds'] = 0
df['no_icds_no_meds'] = 0

df['icds_meds'][(df['n_meds_total']>0) & (df['n_icds'] > 0)] = 1

df['icds_no_meds'][(df['n_meds_total'] == 0) & (df['n_icds'] > 0)] = 1

df['no_icds_meds'][(df['n_meds_total']>0) & (df['n_icds'] == 0)] = 1

df['no_icds_no_meds'][(df['n_meds_total'] == 0) & (df['n_icds'] == 0)] = 1
  
df.to_csv(os.path.join(path,'scores.csv'), index=False)
