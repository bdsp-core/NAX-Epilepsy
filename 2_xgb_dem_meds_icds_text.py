# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:00:28 2025

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

from xgboost import XGBClassifier
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

cols = ['n_icds', 'n_meds', 'Age']

for col in cols:
    X_test[col] = np.round(scale_range (X_test[col], np.min(X_train[col]), np.max(X_train[col])))
    X_test[col] = (X_test[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))
    X_train[col] = (X_train[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))
    
#Pipelines ###################################################################

get_numeric_data = FunctionTransformer(lambda x: x[X_train.columns], validate=False)

clf = XGBClassifier(random_state = 42, max_iter=1000)

num_pipe = Pipeline([
  ('select_num', get_numeric_data),
  ])

full_pipeline = Pipeline([
    ('feat_union', FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipe),
          ])),
    ('clf', clf)
    ])


lst_params = {'clf__n_estimators': [100,150,200,250,300,350],
              'clf__learning_rate': [0.01,0.05, 0.06,0.07,0.08,0.09,0.1],
              'clf__colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8],
              'clf__subsample': [0.8,0.9,1],
              'clf__max_depth': [2,3,4,5],
              'clf__gamma': [0,1,5],
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

# model_name = 'xgb'
# features_name = 'no_prodigy'
# class_name = 'binary'

# ## Best model
# filename = '{}_{}_{}_model.sav'.format(model_name,features_name,class_name)

# # with open(filename, 'wb') as pickle_file:
# #     dill.dump(random_search.best_estimator_, pickle_file)
# #     print('model saved')
    
# # import model
# clf = dill.load(open(filename, 'rb'))
    
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

from performance_binary2 import perf

boot_all_micro, boot_all_macro, boot_label = perf(y_test, y_pred, probs, labels)

#%% Shap #######################################################

X_train = X_train.rename(columns={'n_meds':'number of ASMs',
                                  'n_icds':'number of ICDs',
                                  'with_':'{\'with\', \'epilepsi\'}',
                                  'partial_':'{\'partial\', \'seizur\'}',
                                  'control_':'{\'seizur\', \'control\'}',
                                  'lamotrigin_':'{\'lamotrigin\'},{\'lgt\'}',
                                  'levetiracetam_':'{\'levetiracetam\'},{\'lev\'}',
                                  'breakthrough_':'{\'breakthrough\', \'seizur\'}',
                                  'monotherapy_':'{\'monotherapi\'}',
                                  'generalized_':'{\'general\', \'seizur\'}',
                                  'asneeded_':'{\'follow\', \'up\', \'as\', \'need\'}',
                                  'keppra_':'{\'keppra\'}',
                                  'psychiatric_':'{\'psychiatr\'}',
                                  'depakot_':'{\'depakot\'}',
                                  'trauma_':'{\'trauma\'}',
                                  'focal_':'{\'focal\'}',
                                  'history_':'{\'histori\', \'seizur\'},{\'hx\', \'seizur\'}'
                                  })
import shap

#load JS vis in the notebook
shap.initjs() 

#set the tree explainer as the model of the pipeline
explainer = shap.TreeExplainer(clf.steps[1][1])

#apply the preprocessing to x_test
observations = clf.steps[0][1].transformer_list[0][1][0].transform(X_train)

#get Shap values from preprocessed data
shap_values = explainer.shap_values(observations)


#plot the feature importance
shap.summary_plot(shap_values, X_train, plot_type="bar")

shap.summary_plot(shap_values, X_train)


# Plot bar plot
# shap.plots.bar(explainer(X_train))