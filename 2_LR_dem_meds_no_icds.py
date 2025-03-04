# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 19:35:00 2022

@author: mn46
"""

#%% Libraries #################################################################

import sys
import os
import numpy as np  
import pandas as pd
 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import FunctionTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

# Directory here
path = os.getcwd()  # Get the current working directory
sys.path.insert(0, path) # insert path

###############################################################################
# Test model
#--------------

df = pd.read_csv(os.path.join(path,'icds_encounter_meds_after_build.csv'))


#%% Features for modeling #########################################

outcome = 'outcome'
labels = ['NO', 'YES']


exclude = list(['convulsions seizures','epilepsy and recurrent seizures',
                'syncope','n_icds'])

for i in exclude:
    df = df.drop(columns=i)

df['Sex'] = df.Sex.map({'Male':1, 'Female':0})

#Split patients in train/test #################################################

e = df[['PatientID']].drop_duplicates()

e_train, e_test = train_test_split(e, test_size=0.3, random_state=42) 

# Assign all respective MRNs encounters in train and test
df_train =  df[df.PatientID.isin(e_train.PatientID)] 
df_test =  df[df.PatientID.isin(e_test.PatientID)] 

X_train = df_train.drop(columns=['PatientID','patient_has_epilepsy','Unstructured', 'Date'])
X_test = df_test.drop(columns=['PatientID','patient_has_epilepsy','Unstructured', 'Date'])

y_train = df_train['patient_has_epilepsy'].astype(int)
y_test = df_test['patient_has_epilepsy'].astype(int) 

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

cols = ['Age', 'n_meds']

for col in cols:
    X_test[col] = np.round(scale_range (X_test[col], np.min(X_train[col]), np.max(X_train[col])))
    X_test[col] = (X_test[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))
    X_train[col] = (X_train[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))

#Pipelines ###################################################################

get_numeric_data = FunctionTransformer(lambda x: x[X_train.columns], validate=False)

clf = LogisticRegression(random_state = 42, max_iter=1000)

num_pipe = Pipeline([
  ('select_num', get_numeric_data),
  ])

full_pipeline = Pipeline([
    ('feat_union', FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipe),
          ])),
    ('clf', clf)
    ])


# LR with text
lst_params =  {
                                                                  'clf__C':[0.0001,0.001,0.01,0.1,1.0],
                                                                  'clf__solver':['liblinear', 'lbfgs'],
                                                                  'clf__warm_start':[True,False]}

random_search = RandomizedSearchCV(full_pipeline, param_distributions=lst_params, n_iter=100, cv=5, refit = True, n_jobs=-1, verbose=1, random_state = 42)

#------------------------------------------------------------------------
# Train and test
#------------------------------------------------------------------------

import sys
sys.setrecursionlimit(10000)

random_search.fit(X_train, y_train) 

clf = random_search.best_estimator_

# random_search.best_params_
    
#------------------------------------------------------------------------
# Performance
#------------------------------------------------------------------------

y_train_pred = clf.predict_proba(X_train*1)[:,1]
y_test_pred = clf.predict_proba(X_test*1)[:,1]

#%% Threshold functions #######################################################
 

# Optimal Threshold for Precision-Recall Curve (Imbalanced Classification)
    
def optimal_threshold_auc(target, predicted):
    precision, recall, threshold = precision_recall_curve(target, predicted)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    return threshold[ix]
  
# Threshold in train
threshold = optimal_threshold_auc(y_train, y_train_pred)

y_pred = (clf.predict_proba(X_test*1)[:,1] >= threshold).astype(int)

probs = clf.predict_proba(X_test)

# model_name = 'lr_meds_dem_text'
# features_name = 'no_prodigy'
# class_name = 'binary'

# ## Best model
# filename = '{}_{}_{}_model.sav'.format(model_name,features_name,class_name)

# with open(filename, 'wb') as pickle_file:
#     dill.dump(random_search.best_estimator_, pickle_file)
#     print('model saved')
    
# # import model
# clf = dill.load(open(filename, 'rb'))

from performance_binary2 import perf

boot_all_micro, boot_all_macro, boot_label = perf(y_test, y_pred, probs, labels)

###############################################################################
