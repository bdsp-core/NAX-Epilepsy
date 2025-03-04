# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:46:45 2022

@author: mn46
"""

#%% Libraries #################################################################

import sys
import os 
import pandas as pd


path = os.getcwd()  # Get the current working directory
sys.path.insert(0, path) # insert path

from meds_function import meds_fnc
from icds_function import icds_fnc
from notes_function import notes_fnc

#%% Import data ###############################################################

# Dataset must contain 'PatientID', 'Date', 'Age' at this date, 'Sex'
# The 'Date' column corresponds to Date of visit for each patient in the cohort
# The final dataset has one entry per PatientID with Date of visit

d = pd.read_csv(os.path.join(path,'dataset_demographics.csv'))

#######################
# Add meds
#---------------------

meds = pd.read_csv(os.path.join(path,'meds_all.csv'))
 
ref = pd.read_excel(os.path.join(path, 'meds_reference.xlsx')) 

groups = pd.read_excel(os.path.join(path,'meds_all_grouped.xlsx'))

meds = meds_fnc(d, meds, ref, groups, path)

#######################
# Add icds
#---------------------

icd0 = pd.read_csv(os.path.join(path,'enc_diag_all.csv'), sep=',')

icd1 = pd.read_csv(os.path.join(path,'problist_diag_all.csv'), sep=',')

icd0 = icd0.rename(columns={'ContactDTS':"Date"})
icd1 = icd1.rename(columns={'DiagnosisDTS':"Date"})

icd0.Date = icd0.Date.astype("datetime64[ns]")
icd1.Date = icd1.Date.astype("datetime64[ns]")

cs = ['PatientID', 'Date', 'DiagnosisID']

icds = pd.merge(icd0[cs],icd1[cs],on=['PatientID','Date','DiagnosisID'], how='outer').drop_duplicates()

g = pd.read_excel(os.path.join(path,'icds_grouped.xlsx'))

icds = icds_fnc(icds, d, g, path)

############################################
# Join demographics, meds, icds and notes
#------------------------------------------

meds = pd.read_csv(os.path.join(path,'dataset_meds.csv'), sep=',', index_col=0)

icds = pd.read_csv(os.path.join(path,'dataset_icds_encounter.csv'), sep=',')

meds['Date'] = meds['Date'].astype("datetime64[ns]")
icds['Date'] = icds['Date'].astype("datetime64[ns]")

df = pd.merge(meds, icds, on =['PatientID', 'Date', 'Sex', 'Age', 'patient_has_epilepsy']).drop_duplicates()

#######################
# Add Notes
#---------------------

# # python version # 3.7
# import nltk # 3.6.7
# import sklearn # 0.24.2
# import dill # 0.3.3

# print('The nltk version is {}.'.format(nltk.__version__))
# print('The scikit-learn version is {}.'.format(sklearn.__version__))
# print('The dill version is {}.'.format(dill.__version__))

notes = pd.read_csv(os.path.join(path,'notes.csv'))

col = 'Unstructured'

df_notes = notes_fnc(df, notes, col, path)

df_notes.to_csv(os.path.join(path,'icds_encounter_meds_before_build.csv'), index=False)

###############################################################################
