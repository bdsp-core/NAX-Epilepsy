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

def meds_fnc(d, meds, ref, groups, path):
    
    sys.path.insert(0, path) # insert path
    
    asms = [' Levetiracetam', 'Keppra', 'Lacosamide', 'Vimpat',
    'Phenobarbital', 'Phenytoin', 'Dilantin', 'Valproic acid', 'Valproate', 
    'Depakote', 'Depakene', 'Zonisamide', 'Zonegran', 'Perampanel', 'Fycompa',
    'Clobazam', 'Onfi', 'Clonazepam', 'Klonopin', 'Diazepam','Valium', 
    'Lorazepam','Ativan', 'Oxcarbazepine','Trileptal','Oxtellar',
      'Carbamazepine','Tegretol','Eslicarbazepine','Aptiom',
    'Gabapentin','Neurontin','Pregabalin','Lyrica','Brivaracetam','Briviact', 
      'Cannabidiol','Epidiolex','Cenobamate','Xcopri','Lamotrigine','Lamictal',
    'fosphenytoin','Propofol','midazolam','ketamine','pentobarbital',
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
    
    for item in range(len(asms)):
        asms[item] = asms[item].lower()
        
    # asms
    # ref = pd.read_excel(os.path.join(path+ 'meds_reference.xlsx'))  
    ref.SimpleGenericDSC = ref.SimpleGenericDSC.astype(str).str.lower()
    
    asms[0] = 'levetiracetam'
    asms[len(asms)-1] = 'vigadrone'
        
    ref = ref[(ref.SimpleGenericDSC.astype(str).str.contains('|'.join(asms)))]
    
    def medsfnc(d, ref, aeds):
        
        d.MedicationDSC = d.MedicationDSC.str.lower() # converts to lower case 
        d.MedicationDisplayNM = d.MedicationDisplayNM.str.lower() # converts to lower case 
        d.AmbulatoryMedicationNM = d.AmbulatoryMedicationNM.str.lower() # converts to lower case 
    
        d=d[(d.MedicationDSC.astype(str).str.contains('|'.join(aeds))) | 
            (d.MedicationDisplayNM.astype(str).str.contains('|'.join(aeds))) | 
            (d.AmbulatoryMedicationNM.astype(str).str.contains('|'.join(aeds))) |
            (d.MedicationID.isin(ref.MedicationID))]
        
        return d
    
    #asms
    asms[0] = ' levetiracetam'
    asms[len(asms)-1] = 'vigadrone '
    meds = medsfnc(meds, ref, asms)
    
    # ------------------------------------------------------
    # Dates
    # ------------------------------------------------------
    
    ############################################################################
    # In case we have column Date and not PatientEncounterDateRealNBR, run:
    # meds['Date'] = meds['Date'].astype('datetime64ns')
    
    meds['Date'] = meds.PatientEncounterDateRealNBR.astype(float).apply(lambda x: int(x))
    
    from datetime import datetime
    import time
    from datetime import date, timedelta
    
    StartDate = "12/31/1840"
    
    Date_1 = datetime.strptime(StartDate, "%m/%d/%Y")
    
    meds['Date'] = meds['Date'].apply(lambda x: Date_1 + timedelta(days=x)) 
    
    ############################################################################
    
    meds = meds[['PatientID', 'Date', 'OrderDTS', 'OrderInstantDTS', 
                'OrderStartDTS', 'OrderEndDTS', 'OrderDiscontinuedDTS', 
                'MedicationID', 'OrderStatusDSC',
                'AdditionalInformationOrderStatusDSC', 
                'PendedOrderRefusalReasonDSC', 'MedicationDiscontinueReasonDSC']].drop_duplicates().reset_index().drop(columns='index')
                
    
    meds = meds[meds.OrderStatusDSC.astype(str) != 'Canceled']
    
    meds.OrderStartDTS[meds.OrderStartDTS.astype(str) == 'nan'] = meds.OrderDTS[meds.OrderStartDTS.astype(str) == 'nan']
    
    meds.OrderEndDTS[meds.OrderEndDTS.astype(str) == 'nan'] = meds.OrderDiscontinuedDTS[meds.OrderEndDTS.astype(str) == 'nan']
    
    meds = meds[~((meds.OrderEndDTS.astype(str) == 'nan') & (meds.AdditionalInformationOrderStatusDSC.astype(str) != 'Active Medication'))]
    
    meds = meds[['PatientID','MedicationID','OrderStartDTS','OrderEndDTS']].drop_duplicates()
    
    meds_ = pd.merge(d[['PatientID','Date']].drop_duplicates(), meds, on='PatientID')
    
    meds_ = meds_[meds_.Date >= meds_.OrderStartDTS.astype("datetime64[ns]")]
    
    meds_1 = meds_[(meds_.OrderEndDTS.astype(str) == 'nan')]
    
    meds_2 = meds_[(meds_.Date <= meds_.OrderEndDTS.astype("datetime64[ns]")) & (meds_.OrderEndDTS.astype(str) != 'nan')]
    
    meds = pd.concat([meds_1,meds_2], axis = 0)
    
    
    # group meds
    
    # groups = pd.read_excel(os.path.join(path,'meds_all_grouped.xlsx'))
    
    groups = groups[groups.MedicationID.isin(meds.MedicationID)]
    
    groups = groups[['MedicationID', 'Group into ']].drop_duplicates()
    
    groups['meds_group'] = groups['Group into ']
    
    meds = pd.merge(meds, groups[['MedicationID', 'meds_group']], on='MedicationID', how='inner')
    
    # number of group meds per encounter
    
    a = meds[['PatientID','Date','meds_group']].drop_duplicates()
    
    a = a.groupby(['PatientID','Date']).meds_group.count().reset_index()
    
    a['n_meds'] = a['meds_group']
    
    meds = pd.merge(meds, a[['PatientID','Date','n_meds']], on =['PatientID','Date'], how='inner')
    
    # create cols for each med
    data = meds['meds_group']
    a = pd.get_dummies(data, prefix=None, prefix_sep='_')
    
    meds = pd.concat([meds, a], axis = 1)
    
    aux = meds.groupby(['PatientID', 'Date'])[a.columns].sum()
    
    meds = pd.merge(meds[['PatientID', 'Date','n_meds']],aux, on=['PatientID', 'Date']).drop_duplicates()
    
    # meds 1/0
    
    for col in a.columns:
        meds[col][meds[col] != 0] = 1
    
    df = pd.merge(d, meds, on=['PatientID','Date'], how='outer')
    
    for col in df.columns:
        df[col][df[col].astype(str) == 'nan'] = 0
    
    
    df.to_csv(os.path.join(path,'dataset_meds.csv'), sep=',')
    
    return df
