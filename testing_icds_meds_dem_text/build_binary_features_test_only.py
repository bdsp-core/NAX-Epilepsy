# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:03:31 2022

@author: cdac
"""


import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer


def build_matrix_features(n, col):

    antiEpilepsyBagOfWords = {'evid': {'not', 'evid', 'diagnosi', 'epilepsi'},
                          'recommend': {'not', 'recommend', 'antiepilept', 'medic'},
                          'defer sz': {'defer', 'anti', 'seizur'},
                          'defer med': {'defer', 'anti', 'epilept'},
                          'refer': {'referr', 'gener', 'neurolog'},
                          'follow up': {'not', 'requir', 'follow', 'up'},
                          'followup': {'not', 'requir', 'followup'},
                          'cannot': {'cannot', 'event', 'epilept'},
                          'pnes': {'pnes'},
                          'nosz': {'no', 'seizur', 'event'},
                          'unlikely': {'unlik', 'seizur'},
                          'fnd': {'function', 'neurolog', 'disord'},
                          'migraine': {'migrain'},
                          'anxiety': {'anxieti'},
                          'syncope': {'syncop'},
                          'cd': {'convers', 'disord'},
                          'psycho': {'psychogen'},
                          'risk': {'not', 'have', 'seizur', 'risk', 'factor'},
                          'sleep': {'sleep', 'disord'},
                          'apnea': {'sleep', 'apnea'},
                          'test': {'not', 'recommend', 'test'},
                          'suspicion': {'low', 'suspicion', 'seizur'},
                          'tremor': {'physiolog', 'tremor'},
                          '"seizures"': {"''", 'seizur'},
                          'fn': {'function', 'neurolog'},
                          'vasovagal': {'vasovag'},
                          'pcp': {'defer', 'primary', 'care', 'physician'},
                          'definition': {'not', 'meet', 'definit', 'epilepsi'},
                          'support': {'not', 'support', 'diagnosi', 'epilepsi'},
                          'amnesia': {'amnesia'},
                          'provoke': {'provok', 'seizur'},
                          'depression': {'dispress'},
                          'shiver': {'shiver'},
                          'arrest': {'cardiac', 'arrest'},
                          'noanti': {'no', 'anti', 'seizur', 'medic'},
                          'neuropathy': {'neuropathi'},
                          'neuropathic': {'neuropath'},
                          'meningioma': {'me ningioma'},
                          'holdoff': {'hold', 'off', 'start', 'anti', 'epilept'},
                          'diabetes': {'diabet'},
                          'neurosarcoidosis': {'neurosarcoidosi'},
                          'sdh': {'sdh'},
                          'postoper': {'post', 'oper'},
                          'hemorrhage': {'traumat', 'hemorrhag'},
                          'concern': {'low', 'concern', 'seizur'},
                          'noconcern': {'no', 'concern', 'seizur'},
                          'convince': {'not', 'convinc', 'seizur'},
                          'follow': {'not', 'need', 'follow', 'epilepsi'},
                          'notfollowup': {'not', 'need', 'followup'},
                          'start': {'not', 'start', 'antiepilept', 'medic'},
                          'startsz': {'not', 'start', 'antiseizur', 'medic'},
                          'cause': {'unlik', 'epilepsi'},
                          'trauma': {'trauma'},
                          'traumatic': {'traumat'},
                          'hematoma': {'hematoma'},
                          'abscess': {'brain', 'abscess'},
                          'hold': {'hold', 'off', 'medic'},
                          'postop': {'postop'},
                          'single': {'singl', 'seizur'},
                          'singlesz': {'singl', 'sz'},
                          'funcevents': {'function', 'event'},
                          'asneeded': {'follow', 'up', 'as', 'need'},
                          'asneededfollow': {'followup', 'as', 'need'},
                          'referpsy': {'referr', 'psychiatri'},
                          'defermed': {'defer', 'medic'},
                          'acute': {'acut', 'symptomat', 'seizur'},
                          'symptomatic': {'symptomat', 'seizur'},
                          'first': {'first', 'time', 'seizur'},
                          'lifetime': {'one', 'lifetim', 'seizur'},
                          'evidence': {'no', 'evid', 'seizur'},                          
                          'meet': {'not', 'meet', 'epilepsi'},
                          'notneedmedic': {'not', 'need', 'medic'},
                          'jacobsen': {'jacobsen', 'syndrom'},
                          'alcohol': {'excess', 'alcohol'},
                          'exam': {'normal', 'neurolog', 'exam'},
                          'mri': {'normal', 'mri'},
                          'eeg':{'normal', 'eeg'},
                          'eprisk': {'no', 'epilepsi', 'risk'},
                          'factors': {'no', 'epilepsi', 'risk', 'factor'},
                          'epileptiform': {'no', 'epileptiform', 'abnorm'},
                          'psychiatric': {'psychiatr'},
                          'fentanyl': {'fentanyl'},
                          'bipolar': {'bipolar'},
                          'not have': {'not', 'have', 'epilepsi'},
                          'bite': {'no', 'bite'},
                          'incontinence': {'no', 'incontin'},
                          'lowthres': {'low', 'seizur', 'threshold'},
                          'lowerthres': {'lower', 'seizur', 'threshold'},
                          'antisz': {'no', 'antiseizur', 'medic'},
                          'had': {'not', 'had', 'seizur'},
                          'nonepileptic': {'nonepilept'},
                          'chemo': {'chemo'},
                          'chemotherapy': {'chemotherapi'},
                          'epileptogenic': {'no', 'epileptogen', 'abnorm'},
                          'numb': {'numb'},
                          'surgery': {'surgeri'},
                          'discharge': {'discharg', 'epilepsi', 'clinic'},
                          'nonepileptiform': {'nonepileptiform'},
                          'non epileptiform': {'non', 'epileptiform'},
                          'not epileptic': {'not', 'epilept'},
                          'dementia': {'dementia'},
                          'think': {'not', 'think', 'epilepsi'},
                          'diagnose': {'no', 'diagnosi', 'epilepsi'},
                          'tingling': {'tingl'},
                          'activity': {'not', 'epileptiform', 'activ'},
                          'noseizure': {'no', 'seizur'},
                          'withdrawal': {'withdraw', 'seizur'},
                          'dizzy': {'dizzi'},
                          'maintain': {'maintain', 'conscious'},
                          'electrograph': {'no', 'electrograph', 'seizur'},
                           'wean': {'wean', 'off'},
                           'taper': {'taper'},
                          'resect': {'resect'},
                          'second': {'second', 'opinion'},
                          'definite': {'definit', 'diagnosi', 'epilepsi'},
                          'pseudoseizure': {'pseudoseizur'},
                           'cardiology': {'cardiolog'},
                           'againstsz': {'against', 'seizur'},
                           'against': {'against', 'epilepsi'},
                          'ptsd': {'ptsd'},
                          'pneslong': {'psychogen', 'nonepilept', 'seizur'},
                          'presyncope': {'presyncop'},
                          'hypoglycemia': {'hypoglycemia'},
                          'doubt': {'doubt', 'seizur'},
                          'carry': {'not', 'carri', 'diagnosi', 'epilepsi'},
                          'acutesz': {'acut', 'seizur'},
                          'deny': {'deni', 'seizur'},
                          'spell': {'provok', 'spell'},
                          'non epileptic': {'non', 'epilept', 'spell'},
                          'non  epileptic': {'nonepilept', 'spell'},
                          'insomnia': {'insomnia'},
                          'migraine aura': {'migrain', 'aura'},
                          'clinical': {'no', 'clinic', 'seizur'},
                          'criteria': {'not', 'criteria', 'epilepsi'}}

    proEvidences = {'both': {'both', 'epilepsi', 'pnes'},
                  'mixed dis': {'mix', 'disord'},
                  'ictal': {'ictal'},
                'aura': {'aura'},
                'convulse': {'convuls'},
                'breakthrough': {'breakthrough', 'seizur'},
                'focal': {'focal'},
                'idiopathic': {'idiopath', 'general', 'epilepsi'},
                'history': {'histori', 'seizur'},
                'hx': {'hx', 'seizur'},
                'complex': {'complex', 'seizur'},
                'partial': {'partial', 'seizur'},
                'myoclonic': {'myoclon'},
                'generalized': {'general', 'seizur'},
                'continue': {'continu', 'on'},
                'drive': {'drive', 'month'},
                'szdrive': {'drive', 'seizur'},
                'deja': {'deja', 'vu'},
                'seizurefree': {'seizurefre'},
                'szfree': {'szfree'},
                'seizure free': {'seizur', 'free'},
                'sz free':{'sz', 'free'},
                'frontallobe': {'frontal', 'lobe'},
                'nocturnal': {'nocturn'},
                'febrile': {'febril'},
                'perinatal': {'perinat', 'complic'},
                'control': {'seizur','control'}, 
                'monotherapy': {'monotherapi'},
                'absence': {'absenc', 'seizur'},
                'dejavu': {'dejavu'},
                'postictal': {'postict', 'confus'},
                'tonicclonic': {'tonniclon'},
                'tonic clonic': {'tonic', 'clonic'}, 
                'sudden': {'sudden', 'unexpect', 'death'},
                'sudep': {'sudep'},
                'droop': {'facial', 'droop'},
                'intractable': {'intract', 'epilepsi'},
                'daily': {'daili', 'seizur'},
                'decreased': {'decreas', 'seizur'},
                'device': {'devic'},
                'surgical': {'surgic', 'intervent'},
                'reprogram': {'reprogram'},
                'abnormaleeg': {'abnorm', 'eeg'},
                'with': {'with', 'epilepsi'},
                'juvenile': {'juvenil', 'epilespi'},
                'myoclonus': {'myoclonus'},
                'recurrent': {'recurr', 'sz'},
                'recurrents': {'recurr', 'seizur'},
                'noncompliance': {'noncompli'},
                # 'szdisorder': {'seizur', 'disord'},
                'stable': {'seizur', 'stabl'},
                'shoulder': {'disloc', 'shoulder'},
                'narcolepsy': {'narcolepsi'},
                'sleep clinic': {'sleep', 'clinic'}}
                
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
         'depakot', 'vigabatrin', 'sabril', 'vigadron', 'zonisamid', 'zonegran', 'xanax', 'cocaine']

    matrix = pd.DataFrame()
    names = list(antiEpilepsyBagOfWords.keys()) + list(proEvidences.keys()) + aeds
    columns = list()
       
    matrix[names] = 0 

    stemmer = SnowballStemmer(language='english')
    
    # n = pd.read_csv('C:/Users/cdac/Prodigy/No_ground_truth_1000_cases_epilepsy_clinic.csv')
    n = n.reset_index().drop(columns='index')
    #n = n.dropna()
    n[col] = n[col].str.lower()
    n[col] = n[col].apply(lambda x: " ".join(x.split()))
    # n = n[n.patient_has_epilepsy != 'US']
    n = n.reset_index(drop=True)
    #n['patient_has_epilepsy'] = n.patient_has_epilepsy.map({'YES':2.0, 'NO':0.0, 'US': 1.0})
    # n['patient_has_epilepsy'] = n.patient_has_epilepsy.map({'YES':1, 'NO':0})
    
    # find if a sentence in each of the notes contains one of the key bags of words
    # if it does, add the note to the no epilepsy list
    for index in range(len(n)):
        print(index)
        note = n.iloc[index].get(col)
        note = str(note)
        sentences = sent_tokenize(note)
        foundSentences = set()
        matrix.loc[len(matrix)] = 0
        for sentence in sentences:
            words = word_tokenize(sentence)
            stem_words = []

            for w in words:
                x = stemmer.stem(w)
                stem_words.append(x)

            for word in stem_words:
                if word in aeds:
                    matrix.loc[len(matrix)-1][word] = 1

            for bag in antiEpilepsyBagOfWords:
                if antiEpilepsyBagOfWords[bag].issubset(stem_words):
                    matrix.loc[len(matrix)-1][bag] = 1
                    
            for bag in proEvidences:
                if proEvidences[bag].issubset(stem_words):
                    matrix.loc[len(matrix)-1][bag] = 1
    
    #join like columns together
    def join_columns(df, col1, col2):
        df[col1] = df[col1] + df[col2]
        df = df.drop(columns=[col2])
        return df
    
    matrix = join_columns(matrix, 'history', 'hx')
    matrix = join_columns(matrix, 'follow up', 'followup')
    matrix = join_columns(matrix, 'sz free', 'szfree')
    matrix = join_columns(matrix, 'seizure free', 'seizurefree') 
    matrix = join_columns(matrix, 'sz free', 'seizure free') 
    matrix = join_columns(matrix, 'carbamazepin', 'cbz')
    matrix = join_columns(matrix, 'lamotrigin', 'ltg')
    matrix = join_columns(matrix, 'levetiracetam', 'lev')
    matrix = join_columns(matrix, 'oxcarbazepin', 'oxc')
    matrix = join_columns(matrix, 'topamax', 'tpm')
    matrix = join_columns(matrix, 'lowthres', 'lowerthres')
    matrix = join_columns(matrix, 'chemo', 'chemotherapy')
    matrix = join_columns(matrix, 'epileptiform', 'epileptogenic')
    matrix = join_columns(matrix, 'deja', 'dejavu')
    matrix = join_columns(matrix, 'tonic clonic', 'tonicclonic')
    matrix = join_columns(matrix, 'nonepileptiform', 'non epileptiform')
    matrix = join_columns(matrix, 'asneeded', 'asneededfollow')
    matrix = join_columns(matrix, 'recurrent', 'recurrents')
    matrix = join_columns(matrix, 'single', 'singlesz')
    matrix = join_columns(matrix, 'non epileptic', 'non  epileptic')

    # no_features = matrix.loc[(matrix.sum(axis=1) == 0),]
    # no_features = no_features.join(n.Unstructured)
    # no_features = no_features['Unstructured']
    # no_features.to_csv('C:/Users/cdac/Prodigy/no_ground_truth_no_features_review.csv')

    matrix = matrix.add_suffix('_')
    matrix = matrix.loc[(matrix.sum(axis=1) != 0), (matrix != 0).any(axis=0)]
    
    # df_all = n.merge(matrix.drop_duplicates(), on=['PatientID', 'Date'], 
    #                how='left', indicator=True)
    # n['merge'] = df_all['_merge']
    # df = n[n['merge'] == 'both']        
    #n = n[n.index.isin(matrix.index)] # same indexes
    # matrix = pd.concat([matrix, n.patient_has_epilepsy], axis = 1)
    # matrix = pd.concat([matrix, n.PatientID], axis = 1)
    # matrix = pd.concat([matrix, n.Unstructured], axis = 1)
    matrix = pd.concat([matrix, n[n.columns]], axis=1)

    matrix = matrix.reset_index().drop(columns='index')
    
    return matrix
