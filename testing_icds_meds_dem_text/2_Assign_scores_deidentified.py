# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:10:02 2023

@author: mn46
"""


#%% Libraries #################################################################

import sys
import os
import re
import nltk
import numpy as np  
import pandas as pd


# Directory here
path = 'C:/Users/mn46/Downloads/Test_epilepsy_alg/testing_icds_meds_dem_text/'
sys.path.insert(0, path) # insert path

###############################################################################
# Test model
#--------------

# df_scores = assign_scores(df_test, df, col_notes, path)

from sklearn.model_selection import train_test_split
import dill

df_test = pd.read_csv(os.path.join(path,'df_final.csv'))

# Import reference training data

X_train = pd.read_csv(os.path.join(path, 'X_train.csv'))

y_train = pd.read_csv(os.path.join(path, 'y_train.csv'))

#%% Features for modeling #########################################

outcome = 'outcome'
labels = ['NO', 'YES']

df_test['Sex'][df_test['Sex'].astype(str) == 'Male'] = 1
df_test['Sex'][df_test['Sex'].astype(str) != '1'] = 0
df_test['Sex'] = df_test['Sex'].astype(int)

#Split patients in train/test #################################################

X_test = df_test.drop(columns=['PatientID', 'Date'])


def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

cols = ['n_icds', 'n_meds', 'Age']

for col in cols:
    X_test[col] = np.round(scale_range (X_test[col], np.min(X_train[col]), np.max(X_train[col])))
    X_test[col] = (X_test[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))
    X_train[col] = (X_train[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))

cols_missing = list(X_train.columns[~(X_train.columns.isin(X_test.columns))])

for i in cols_missing:
    X_test[i] = 0

X_test = X_test[list(X_train.columns)]

#------------------------------------------------------------------------
# Load model - check directory folder
#------------------------------------------------------------------------

model_name = 'lr_meds_icds_dem_text_'
features_name = 'no_prodigy'
class_name = 'binary'

## Best model
filename = '{}_{}_{}_model.sav'.format(model_name,features_name,class_name)

# import model
clf = dill.load(open(filename, 'rb'))

#------------------------------------------------------------------------
# Test model
#------------------------------------------------------------------------

y_train_pred = clf.predict_proba(X_train*1)[:,1]
y_test_pred = clf.predict_proba(X_test*1)[:,1]

#%% Threshold functions #######################################################

# Optimal Threshold for Precision-Recall Curve (Imbalanced Classification)

from sklearn.metrics import precision_recall_curve
    
def optimal_threshold_auc(target, predicted):
    precision, recall, threshold = precision_recall_curve(target, predicted)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    return threshold[ix]
  
# Threshold in train
threshold = optimal_threshold_auc(y_train, y_train_pred) 

y_pred = (clf.predict_proba(X_test*1)[:,1] >= threshold).astype(np.int)

probs = clf.predict_proba(X_test)

# Assign scores

df_scores = pd.concat([df_test, pd.DataFrame(probs,columns=['prob_NO','prob_YES'])], axis = 1)
df_scores = pd.concat([df_scores, pd.DataFrame(y_pred, columns=['model_answer'])], axis = 1)

df_scores['Sex'][df_scores['Sex'] == 1] = 'Male'
df_scores['Sex'][df_scores['Sex'] == 0] = 'Female'
    
df_scores.to_csv(os.path.join(path,'dataset_with_scores.csv'), index=False)
###############################################################################