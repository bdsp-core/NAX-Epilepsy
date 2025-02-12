# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:10:11 2022

@author: mn46
"""


import sys
import os
import re
import numpy as np  
import pandas as pd

def meds_fnc(enc, meds, path):
    
    sys.path.insert(0, path) # insert path
    
    asms = [' Levetiracetam', 'Keppra', 'Lacosamide', 'Vimpat',
    'Phenobarbital', 'Phenytoin', 'Dilantin', 'Valproic acid', 'Valproate', 
    'Depakote', 'Depakene', 'Zonisamide', 'Zonegran', 'Perampanel', 'Fycompa',
    'Clobazam', 'Onfi', 'Clonazepam', 'Klonopin', 'Diazepam','Valium', 
    'Lorazepam','Ativan', 'Oxcarbazepine','Trileptal','Oxtellar',
      'Carbamazepine','Tegretol','Eslicarbazepine','Aptiom',
    'Gabapentin','Neurontin','Pregabalin','Lyrica','Brivaracetam','Briviact', 
      'Cannabidiol','Epidiolex','Cenobamate','Xcopri','Lamotrigine','Lamictal',
    'fosphenytoin','Propofol','midazolam','nayzilam','ketamine','pentobarbital',
    'Acetazolamide', 'ACTH', 'Epitol', 'Equetro', 'Carbatrol', 
    'Frisium', 'Sympazan', 'Epitril', 'Rivotril', 'Clorazepate',
    'Tranxene', 'Gen Xene','Diamox', 'Diastat', 'Divalproex Sodium',
    'Ethosuximide','Zarontin','Ethotoin', 'Ezogabine', 'Potiga', 
    'Felbamate', 'Felbatol', 'Gralise','Horizant', 'Roweepra', 'Spritam',
    'Elepsia XR', 'Methsuximide','Methosuximide', 'Celontin',
    'Luminol', 'Luminal', 'Epanutin', 'Phenytek', 'Primidone', 'Mysoline',
    'Rufinamide', 'Banzel', 'Inovelon', 'Stiripentol', 'Diacomit',
    'Tiagabine', 'Gabitril', 'Topiramate', 'Topamax', 'Qudexy XR', 
    'Trokendi XR', 'Convulex', 'Depacon', 'Depakene', 'Orfiril', 
    'Valporal', 'Valprosid', 'Vigabatrin', 'Sabril', 'Vigadrone ']
    
    # 'nayzilam' added
    
    for item in range(len(asms)):
        asms[item] = asms[item].lower()
        
    # asms
    
    asms[0] = 'levetiracetam'
    asms[len(asms)-1] = 'vigadrone'
        
    def medsfnc(d, aeds):
        
        d.MedicationDSC = d.MedicationDSC.str.lower() # converts to lower case 
        d.MedicationDisplayNM = d.MedicationDisplayNM.str.lower() # converts to lower case 
        d.AmbulatoryMedicationNM = d.AmbulatoryMedicationNM.str.lower() # converts to lower case 
    
        d=d[(d.MedicationDSC.astype(str).str.contains('|'.join(aeds))) | 
            (d.MedicationDisplayNM.astype(str).str.contains('|'.join(aeds))) | 
            (d.AmbulatoryMedicationNM.astype(str).str.contains('|'.join(aeds)))]
        
        return d
    
    #asms
    asms[0] = ' levetiracetam'
    asms[len(asms)-1] = 'vigadrone '
    meds = medsfnc(meds, asms)
    
    # group meds
    
    def group_meds(meds, col):
        
        # order of the asms is important
        l = ['acetazolamide|diamox',
         'eslicarbazepine|aptiom',
         'lorazepam|ativan',
         'rufinamide|banzel',
         'brivaracetam|briviact',
         'cannabidiol|epidiolex',
         'carbamazepine|carbatrol|epitol|equetro|tegretol',
         'methsuximide|celontin',
         'cenobamate|xcopri',
         'clobazam|onfi|sympazan',
         'clonazepam|klonopin',
         'clorazepate|tranxene',
         'valproic acid|depacon|depakene|depakote|divalproex|stavzor|valproate',
         'diazepam|diastat|valium|valtoco',
         'fosphenytoin',
         'mephenytoin',
         'phenytoin|dilantin|phenytek',
         'levetiracetam|elepsia|keppra|roweepra|spritam',
         'esketamine|spravato',
         'ethosuximide|zarontin',
         'ezogabine|potiga',
         'gabapentin|fanatrex|gabacaine|gralise|horizant|neurontin|smartrx',
         'felbamate|felbatol',
         'perampanel|fycompa',
         'tiagabine|gabitril',
         'midazolam|nayzilam',
         'ketamine|ketalar',
         'lacosamide|vimpat',
         'lamotrigine|lamictal|subvenite',
         'phenobarbital|luminal',
         'pregabalin|lyrica',
         'primidone|mysoline',
         'oxcarbazepine|oxtellar|trileptal',
         'ethotoin|peganone',
         'phenobarbital|pentobarbital',
         'propofol',
         'topiramate|qudexy|topamax|topiragen|trokendi',
         'vigabatrin|sabril',
         'zonisamide|zonegran']
       
        meds[col] = meds[col].astype(str).str.lower()
       
        meds[col][( (meds[col].astype(str).str.contains('phentermine')) &
                               (meds[col].astype(str).str.contains('topiramate')) ) |
                               meds[col].astype(str).str.contains('qsymia')] = 'phentermine/topiramate'
       
        meds[col][( (meds[col].astype(str).str.contains('propantheline')) &
                               (meds[col].astype(str).str.contains('phenobarbital')) )] = 'propantheline/phenobarbital' 
          
        for i in l:
        
            find = re.compile(r"^[^|]*")
           
            if i == 'ketamine|ketalar':
                meds[col][ (meds[col].astype(str).str.contains(i))
                                   & (meds[col].astype(str) != 'esketamine')] = re.search(find, i).group(0)
         
            elif i == 'phenytoin|dilantin|phenytek':
                meds[col][ (meds[col].astype(str).str.contains(i))
                                    & (meds[col].astype(str) != 'mephenytoin')
                                    & (meds[col].astype(str) != 'fosphenytoin')
                                    ] = re.search(find, i).group(0)
            else:    
                meds[col][ (meds[col].astype(str).str.contains(i))
                                       & (meds[col].astype(str) != 'phentermine/topiramate')
                                       & (meds[col].astype(str) != 'propantheline/phenobarbital')
                                       ] = re.search(find, i).group(0)
        return meds
      
        
    cols = ['MedicationDSC','MedicationDisplayNM','AmbulatoryMedicationNM']
    
    for i in cols:
        meds = group_meds(meds, i)
    
    # ------------------------------------------------------
    # Dates
    # ------------------------------------------------------
    
    
    meds = meds[['PatientID', 'MedicationDSC', 'OrderDTS', 'OrderStartDTS', 'OrderEndDTS', 
                 'OrderDiscontinuedDTS', 'OrderStatusDSC',
                'AdditionalInformationOrderStatusDSC']].drop_duplicates().reset_index().drop(columns='index')
                
    meds = meds[meds.OrderStatusDSC.astype(str) != 'Canceled']
    
    meds.OrderStartDTS[meds.OrderStartDTS.astype(str) == 'nan'] = meds.OrderDTS[meds.OrderStartDTS.astype(str) == 'nan']
    
    meds.OrderEndDTS[meds.OrderEndDTS.astype(str) == 'nan'] = meds.OrderDiscontinuedDTS[meds.OrderEndDTS.astype(str) == 'nan']
    
    meds = meds[~((meds.OrderEndDTS.astype(str) == 'nan') & (meds.AdditionalInformationOrderStatusDSC.astype(str) != 'Active Medication'))]
    
    meds = meds[['PatientID','OrderStartDTS','OrderEndDTS','MedicationDSC']].drop_duplicates()
    
    
    meds_ = pd.merge(enc[['PatientID','ContactDTS']].drop_duplicates(), meds, on='PatientID')
    
    meds_ = meds_[meds_['ContactDTS'] >= meds_['OrderStartDTS'].astype("datetime64[ns]")]
    
    meds_1 = meds_[(meds_['OrderEndDTS'].astype(str) == 'nan')]
    
    meds_2 = meds_[(meds_['ContactDTS'] <= meds_['OrderEndDTS'].astype("datetime64[ns]")) & (meds_['OrderEndDTS'].astype(str) != 'nan')]
    
    meds = pd.concat([meds_1,meds_2], axis = 0)
    
  
    # number of group meds per encounter
    
    a = meds[['PatientID','ContactDTS','MedicationDSC']].drop_duplicates()
    
    a = a.groupby(['PatientID','ContactDTS']).MedicationDSC.count().reset_index()
    
    a['n_meds'] = a['MedicationDSC']
    
    meds = pd.merge(meds, a[['PatientID','ContactDTS','n_meds']], on =['PatientID','ContactDTS'], how='inner')
    
    # create cols for each med
    data = meds['MedicationDSC']
    a = pd.get_dummies(data, prefix=None, prefix_sep='_')
    
    meds = pd.concat([meds, a], axis = 1)
    
    aux = meds.groupby(['PatientID', 'ContactDTS'])[a.columns].sum()
    
    meds = pd.merge(meds[['PatientID', 'ContactDTS','n_meds']],aux, on=['PatientID', 'ContactDTS']).drop_duplicates()
    
    # meds 1/0
    
    for col in a.columns:
        meds[col][meds[col] != 0] = 1
    
    meds = meds.rename(columns={'carbamazepine':'carbamezapine'})
    meds.to_csv(os.path.join(path,'dataset_meds.csv'), index = False, sep=',')
    
    return meds
