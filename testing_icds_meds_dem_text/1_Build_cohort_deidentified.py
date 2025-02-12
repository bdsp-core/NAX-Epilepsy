# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:49:51 2023

@author: mn46
"""

#%% Libraries #################################################################

import sys
import os
import re
import numpy as np  
import pandas as pd
from datetime import datetime, date, timedelta
import time
from dateutil.relativedelta import relativedelta

# Directory here
path = 'C:/Users/mn46/Downloads/Test_epilepsy_alg/testing_icds_meds_dem_text/'
sys.path.insert(0, path) # insert path

from meds_function import meds_fnc
from icds_function import icds_fnc
from notes_function import notes_fnc


#%% Load your dataset with PatientID and reference dates ####################

d = pd.read_csv(os.path.join(path,'PatientsData.csv'))

################################################################
# Reference dates and time windows
#------------------------------------

# ContactDTS is the reference date for outpatient

# HospitalAdmitDTS is the reference date for inpatient

# if LOS: (#uncomment these lines and comment time window lines)
    # d['Date_before'] = d['HospitalAdmitDTS']
    # d['Date_after'] = d['HospitalDischargeDTS']

# if time window (other than LOS):
    # Run the code for time window
        
# Code for time window options:
    # n amount of time_ before and after the reference
    # n amount of time_ only before the reference
    # n amount of time_ only after the reference 
    # input n=0 for HospitalAdmitDTS/ContactDTS same day only (no time window)

################################################################
# Time windows code
#------------------------------------

# Replace ContactDTS by HospitalAdmitDTS if inpatient

# specify time (years/months/days)
time_ = 'years'

# specify amount of time (n=0 gives the reference/same day only)
n = 2

if time_ == 'years':
    
    d['Date_before'] = d['ContactDTS'].astype("datetime64[ns]").apply(lambda x: x - relativedelta(years=n))
    d['Date_after'] = d['ContactDTS'].astype("datetime64[ns]").apply(lambda x: x + relativedelta(years=n))
    
elif time_ == 'months':

    d['Date_before'] = d['ContactDTS'].astype("datetime64[ns]").apply(lambda x: x - relativedelta(months=n))
    d['Date_after'] = d['ContactDTS'].astype("datetime64[ns]").apply(lambda x: x + relativedelta(months=n))

elif time_ == 'days':

    d['Date_before'] = d['ContactDTS'].astype("datetime64[ns]").apply(lambda x: x - relativedelta(days=n))
    d['Date_after'] = d['ContactDTS'].astype("datetime64[ns]").apply(lambda x: x + relativedelta(days=n))

# Uncomment any of these three cases if it applies

# if we want all encounters before ContactDTS/HospitalAdmitDTS only: date after becomes the reference
# d['Date_after'] = d['ContactDTS/HospitalAdmitDTS '].astype("datetime64[ns]")

# if we want all encounters after ContactDTS/HospitalAdmitDTS  only: date before becomes the reference
# d['Date_before'] = d['ContactDTS/HospitalAdmitDTS '].astype("datetime64[ns]")

# if we want all encounters after ContactDTS/HospitalDischargeDTS only: date before becomes the reference
# d['Date_before'] = d['ContactDTS/HospitalDischargeDTS'].astype("datetime64[ns]")


# check if time window is within specific time range that you have data for,
# and assign limit date in case needed (# uncomment and input date)
# d['Date_after'][d['Date_after'] > '2023-05-23'] = '2023-05-23'
# d['Date_after'] = d['Date_after'].astype('datetime64[ns]')


################################################################################
# Encounters dates for the list of patients 
#-------------------------------------------------------------------------------

# Query 1

encounters = pd.read_csv(os.path.join(path,'PatientsData_outpatient.csv')) # HospitalAdmitDTS and HospitalDischargeDTS can be disregarded

# PatientsData_outpatient.csv -- has outpatient visits that may result in hospital admissions (reference date will be ContactDTS)

# PatientsData_inpatient.csv -- has inpatient hospital admissions only (reference date will be HospitalAdmitDTS)

df = pd.merge(d[['PatientID','Date_before','Date_after']], encounters, on =['PatientID'], how='outer').drop_duplicates()

# Uncomment according to the case selected in the beginning, in case it applies

# if we want all encounters before ContactDTS/HospitalAdmitDTS only: date after becomes the reference
# df = df[(df['ContactDTS/HospitalAdmitDTS'] <= df['Date_after'])]

# if we want all encounters after ContactDTS/HospitalAdmitDTS only: date before becomes the reference
# df = df[(df['ContactDTS/HospitalAdmitDTS'] >= df['Date_before'])]

# if we want all encounters after ContactDTS/HospitalDischargeDTS only: date before becomes the reference
# df = df[(df['ContactDTS/HospitalDischargeDTS'] >= df['Date_before'])]

# for time window code (also for same day)
df = df[(df['ContactDTS'].astype('datetime64[ns]') >= df['Date_before'].astype('datetime64[ns]')) & (df['ContactDTS'].astype('datetime64[ns]') <= df['Date_after'].astype('datetime64[ns]'))]

df['BirthDTS'] = df['BirthDTS'].astype('datetime64[ns]')
df['ContactDTS'] = df['ContactDTS'].astype('datetime64[ns]')

df['Age'] = np.nan
df['Age'] = df.apply(lambda x: relativedelta(x['ContactDTS'],x['BirthDTS']).years, axis=1)

# check if BirthDTS is correct


#######################
# Add meds
#---------------------

# Query 2

meds = pd.read_csv(os.path.join(path,'medications.csv'))

meds = meds_fnc(df, meds, path)

#######################
# Add icds
#---------------------

# Query 3 

icds = pd.read_csv(os.path.join(path,'icds.csv'))

# Query 4

probs = pd.read_csv(os.path.join(path,'probList.csv'))

g = pd.read_csv(os.path.join(path,'icds_groups.csv'))

icds = icds_fnc(icds, probs, df, g, path)

############################################
# Add notes
#------------------------------------------

# Query 5

notes = pd.read_csv(os.path.join(path,'notes_example.csv')) # sample of notes as example

notes.ContactDTS = notes.ContactDTS.astype("datetime64[ns]")

notes = pd.merge(d[['PatientID','Date_before','Date_after']], notes[notes.PatientID.isin(d.PatientID)], on =['PatientID'], how='outer').drop_duplicates()

# for time window code (also for same day)
notes = notes[(notes.ContactDTS >= notes.Date_before) & (notes.ContactDTS <= notes.Date_after)]

notes = notes.drop(columns=['Date_before','Date_after'])

col_notes = 'NoteTXT'

df_notes = notes_fnc(notes, col_notes, path) # check inside function 

df_notes = pd.read_csv(os.path.join(path,'matrix_build.csv')) 

df_notes.Date = df_notes.Date.astype("datetime64[ns]")

df_notes = pd.merge(d[['PatientID','Date_before','Date_after']].drop_duplicates(), df_notes[df_notes.PatientID.isin(d.PatientID)], on =['PatientID'], how='outer').drop_duplicates()

df_notes = df_notes[(df_notes.Date >= df_notes.Date_before) & (df_notes.Date <= df_notes.Date_after)]

df_notes = df_notes.drop(columns=['Date_before','Date_after'])

# check for any typos
# df_notes['tegretol_'][df_notes['tegretol_.1'] == 1] = 1
# df_notes['epitol_'][df_notes['epitol_.1'] == 1] = 1
# df_notes['topiram_'][df_notes['topiram_.1'] == 1] = 1
# df_notes['depakot_'][df_notes['depakot_.1'] == 1] = 1

# df_notes = df_notes.drop(columns=['topiram_.1','depakot_.1'])            

############################################
# Join demographics, meds and icds
#------------------------------------------

meds = pd.read_csv(os.path.join(path,'dataset_meds.csv'))

icds = pd.read_csv(os.path.join(path,'dataset_icds.csv'))

i = 'ContactDTS'
meds[i] = meds[i].astype("datetime64[ns]")
icds[i] = icds[i].astype("datetime64[ns]")
df_notes[i] = df_notes['Date'].astype("datetime64[ns]")

dfs = pd.merge(meds, icds, on =['PatientID', 'ContactDTS'], how='outer').drop_duplicates()

dfs = pd.merge(dfs, df_notes.drop(columns='Date'), on =['PatientID', 'ContactDTS'], how='outer').drop_duplicates()

dfs = pd.merge(d[['PatientID','Date_before','Date_after']], dfs[dfs.PatientID.isin(d.PatientID)], on =['PatientID'], how='outer').drop_duplicates()

# Uncomment according to the case selected in the beginning, in case it applies

# if we want all encounters before transferindts only: date after becomes the reference
# dfs = dfs[(dfs['Date'] <= dfs['Date_after'])]

# if we want all encounters after transferindts only: date before becomes the reference
# dfs = dfs[(dfs['Date'] >= dfs['Date_before'])]

# if we want all encounters after transferoutdts only: date before becomes the reference
# dfs = dfs[(dfs['Date'] >= dfs['Date_before'])]

# for time window code (also for same day)
dfs = dfs[(dfs['ContactDTS'] >= dfs['Date_before']) & (dfs['ContactDTS'] <= dfs['Date_after'])]

df = pd.merge(dfs.drop(columns=['Date_before', 'Date_after']), df[['PatientID','ContactDTS','SexDSC','Age']].drop_duplicates(), on =['PatientID','ContactDTS'], how='outer').drop_duplicates()

for i in list(df.columns):
    df[i][df[i].astype(str) == 'nan'] = 0

df = df[df.ContactDTS.astype("str") != 'NaT']
          
df = df.rename(columns={'SexDSC':'Sex','ContactDTS':'Date'}).drop(columns='NoteTXT')
   
 
df.to_csv(os.path.join(path,'df_final.csv'), index=False)

###############################################################################
