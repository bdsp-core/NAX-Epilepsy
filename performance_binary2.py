# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:52:50 2022

@author: mn46
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:39:08 2020

@author: mn46
"""


#%% libraries #############################################################

import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt   
import itertools
from itertools import cycle
from sklearn.metrics import (confusion_matrix, average_precision_score,
                             accuracy_score, recall_score, f1_score, auc,
                             precision_recall_curve, roc_auc_score, roc_curve, 
                             precision_score) #, specificity_score) 

from imblearn.metrics import specificity_score

# cohen_kappa_score,confusion_matrix      

from numpy import median, percentile
from numpy.random import seed, randint


def perf(y_true, y_pred, probs, labels):
        
    ##############################
    # Target and predicted labels
    #---------------------------

    num_labels = len(labels)
    
    # y target
    y_true = np.array(pd.get_dummies(y_true))
    
    y_pred0 = y_pred
    
    # y predicted - hard labels
    y_pred = np.array(pd.get_dummies(y_pred))
    
    if len(probs) !=0:
        y_pred = probs
    
    ###------------------------------------------------------------------------
    ### ROC
    ###------------------------------------------------------------------------
    
    colors = cycle(['steelblue', 'aqua', 'mediumorchid'])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_labels)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_labels):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= num_labels
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    if len(labels) > 5:
        plt.figure(figsize=(12,8))
    else:
        plt.figure(figsize=(12, 10))
    plt.rcParams.update({'font.size': 40})
    # plt.plot(fpr["micro"], tpr["micro"],
    #           label='micro-average AUROC = {0:0.2f})'
    #                 ''.format(roc_auc["micro"]),
    #           color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
              label='macro-average AUROC = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
              color='navy', linestyle=':', linewidth=4)
    
    
    for i, color in zip(range(num_labels), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                  label='{0} (AUROC = {1:0.2f})'
                  ''.format(labels[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('AUROC multi-class - hard labels') #
    # if len(probs) !=0:
    #     plt.title('AUROC multi-class - soft labels')
    if len(labels) > 5:
        plt.legend(loc="lower right",fontsize = 30)
    else:
        plt.legend(loc="lower right",fontsize = 30)
    plt.show()
    
    specificity= dict()
    specificity['micro'] = 1-fpr['micro']
    # sensitivity = tpr['micro']
    
    ###------------------------------------------------------------------------
    ### AUPR
    ###------------------------------------------------------------------------
    
    colors = cycle(['steelblue', 'aqua', 'mediumorchid'])
    
    # For each class
    precision = dict()
    recall = dict() 
    average_precision = dict()
    for i in range(num_labels):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
        y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_pred,
                                                         average="micro")
    
    # A "macro-average": quantifying score on all classes jointly
    precision["macro"], recall["macro"], _ = precision_recall_curve(y_true.ravel(),
        y_pred.ravel())
    average_precision["macro"] = average_precision_score(y_true, y_pred,
                                                         average="macro")
    ###------------------------------------------------------------------------
    ### AUPR - discriminated
    ###------------------------------------------------------------------------
    
    # setup plot details
    plt.figure(figsize=(12, 10))
    plt.rcParams.update({'font.size': 40})
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels_ = []
    # for f_score in f_scores:
    #     x = np.linspace(0.01, 1)
    #     y = f_score * x / (2 * x - f_score)
    #     l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    #     plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.7, y[45] + 0.02))
    
    # lines.append(l)
    #labels_.append('iso-f1 curves')
    # l, = plt.plot(recall["micro"], precision["micro"], color='deeppink', linestyle=':', linewidth=6)
    # lines.append(l)
    l, = plt.plot(recall["macro"], precision["macro"], color='navy', linestyle=':', linewidth=6)
    lines.append(l)
    # labels_.append('micro-average (AP = {0:0.2f})'
    #               ''.format(average_precision["micro"]))
    labels_.append('macro-average (AP = {0:0.2f})'
                  ''.format(average_precision["macro"]))
    
    
    for i, color in zip(range(num_labels), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels_.append('{0} (AP = {1:0.2f})'
                      ''.format(labels[i], average_precision[i]))
    
    # fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title('Precision-Recall curve to multi-class - hard labels')
    # plt.title('Precision-Recall curve to multi-class - soft hot encoded labels')
    # plt.title('Precision-Recall curve to multi-class - soft labels')
    plt.legend(lines, labels_, loc=(0.05, 0.12), fontsize = 30)#prop=dict(size=20))
    plt.show()
    
    #------------------------------------------------------------------------
    # Confusion matrix
    #------------------------------------------------------------------------
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix', n = 'None',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
            
        if normalize:
            if n == 'recall':
                axis = 1
                cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis]
            elif n == 'precision':
                axis = 0
                cm = cm.astype('float') / cm.sum(axis=axis)[np.newaxis,:]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
            
        print(cm)
    
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.rcParams.update({'font.size': 12}) # 18 for GOS, 14 for mRS
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=60)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
       
        
    
    cnf_matrix = confusion_matrix(np.argmax(np.array(y_true), axis=1), y_pred0)
    
    # Plot normalized by recall confusion matrix
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14}) 
    plot_confusion_matrix(cnf_matrix, classes=np.asarray(labels), normalize=True, n = 'recall')
    #plt.grid(b=None)
                           
    # Plot normalized by precision confusion matrix
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14}) 
    plot_confusion_matrix(cnf_matrix, classes=np.asarray(labels), normalize=True, n = 'precision')
    #plt.grid(b=None)
    
    fig = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.asarray(labels), normalize=False, n = 'None')
                         
    #plt.grid(b=None) 
    fig = plt.figure()
    
    

    #%% Bootstrapping function for Multiclass #####################################
    
    
    def get_CI_boot(y_true,y_pred,metric,boot,metric_average):
    
        # bootstrap confidence intervals
        # seed the random number generator
        seed(1)
        i = 0
        # generate dataset
        dataset = y_pred
        real = y_true
        # bootstrap
        scores = list()
        while i < boot:
            # bootstrap sample
            indices = randint(0, len(y_pred) - 1, len(y_pred))
            sample = dataset[indices]
            real = y_true[indices]
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
        	# calculate and store statistic 
            else:
                if (metric == roc_auc_score) & (metric_average == 'micro'): 
                        # Compute micro-average ROC curve and ROC area
                    fpr[metric_average], tpr[metric_average], _ = roc_curve(real.ravel(), sample.ravel())
                    roc_auc[metric_average] = auc(fpr[metric_average], tpr[metric_average])              
                    scores.append(roc_auc[metric_average])
                    i += 1
                    
                elif (metric == roc_auc_score) & (metric_average == 'macro'): 
                      # Compute micro-average ROC curve and ROC area
                     roc_auc_ = roc_auc_score(real, sample)              
                     scores.append(roc_auc_)
                     i += 1 
                
                elif (metric == metric_aupr_macro): 
                      # Compute micro-average ROC curve and ROC area
          
                     precisions, recalls, _ = precision_recall_curve(real.ravel(), sample.ravel()) 
                     # precision1, recall1, _ = precision_recall_curve(real[:,0], sample[:,0]) 
                     # precision2, recall2, _ = precision_recall_curve(real[:,1], sample[:,1]) 
                     # precisions = (precision1 + precision2)/2
                     # recalls = (recall1 + recall2)/2
                     aucs = auc(recalls, precisions)             
                     scores.append(aucs)
                     i += 1 
                     
                elif (metric == metric_aupr_micro): 
                     # Compute micro-average ROC curve and ROC area
                     precision["micro"], recall["micro"], _ = precision_recall_curve(real.ravel(), sample.ravel())
                     aucs = auc(recall["micro"], precision["micro"])   
                     scores.append(aucs)
                     i += 1         
                     
                elif metric == average_precision_score:
                        # A "micro-average": quantifying score on all classes jointly
                    average_precision[metric_average] = average_precision_score(real, sample, average=metric_average)                
                    scores.append(average_precision[metric_average])
                    i += 1
                elif metric == precision_score:
                    precision[metric_average] = precision_score(real, sample, average=metric_average)
                    scores.append(precision[metric_average] )
                    i += 1
                elif metric == recall_score:
                    recall[metric_average] = recall_score(real, sample, average=metric_average)              
                    scores.append(recall[metric_average])
                    i += 1 
                elif (metric == spec) & (metric_average == 'macro'):
                    
                    # def spec(y_true,y_pred):
                    #     TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()
                    #     return TN/(TN+FP)  
                    
                    specificity1 = spec(real[:,0],sample[:,0])
                    specificity2 = spec(real[:,1],sample[:,1])
                    specificity[metric_average] = (specificity1 + specificity2)/2
                                  
                    scores.append(specificity[metric_average])
                    i += 1 
                    
                elif (metric == specificity_score) & (metric_average == 'micro'):
                    specificity[metric_average] = specificity_score(real[:,1], sample[:,1], average=metric_average)              
                    scores.append(specificity[metric_average])
                    i += 1     
                elif metric == accuracy_score:
                    scores.append(accuracy_score(real, sample))
                    i += 1   
                elif metric == f1_score:
                    scores.append(f1_score(real, sample, average=metric_average))
                    i += 1          
        # calculate 95% confidence intervals (100 - alpha)
        alpha = 5.0
        # calculate lower percentile (e.g. 2.5)
        lower_p = alpha / 2.0
        # retrieve observation at lower percentile
        lower = max(0.0, percentile(scores, lower_p))
        # calculate upper percentile (e.g. 97.5)
        upper_p = (100 - alpha) + (alpha / 2.0)
        # retrieve observation at upper percentile
        upper = min(1.0, percentile(scores, upper_p))
        return (lower,upper)
    
    
    #%% Bootstrapping function ###################################################
        
    def get_CI_boot_outcome(y_true,y_pred,metric,boot):
        # bootstrap confidence intervals
        # seed the random number generator
        seed(1)
        i = 0
        # generate dataset
        dataset = y_pred
        real = y_true
        # bootstrap
        scores = list()
        while i < boot:
            # bootstrap sample
            indices = randint(0, len(y_pred) - 1, len(y_pred))
            sample = dataset[indices]
            real = y_true[indices]
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
        	# calculate and store statistic 
            else:
                statistic = metric(real,sample)
                scores.append(statistic)
                i += 1
        # calculate 95% confidence intervals (100 - alpha)
        alpha = 5.0
        # calculate lower percentile (e.g. 2.5)
        lower_p = alpha / 2.0
        # retrieve observation at lower percentile
        lower = max(0.0, percentile(scores, lower_p))
        # calculate upper percentile (e.g. 97.5)
        upper_p = (100 - alpha) + (alpha / 2.0)
        # retrieve observation at upper percentile
        upper = min(1.0, percentile(scores, upper_p))
        return (lower,upper)
    
    
    #%% Bootsrapping results in test multiclass ###################################
    
    def metric_aupr_macro(y_true, y_pred):

        precisions, recalls, _ = precision_recall_curve(real.ravel(), sample.ravel()) 
        aucs = auc(recalls, precisions)
        
        return aucs
    
    def metric_aupr_micro(y_true, y_pred):

        precision["micro"], recall["micro"], _ = precision_recall_curve(real.ravel(), sample.ravel())
        aucs = auc(recall, precision)  
        
        return aucs
   
    
    def spec(y_true,y_pred):
        TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()
        return TN/(TN+FP)   
    
    metrics = [roc_auc_score, accuracy_score,  recall_score, spec,
               f1_score, average_precision_score, precision_score, metric_aupr_macro]
  
    myp = []
    mye = []
    
    boot=1000
    metric_average = "macro"
       
    e = []
    for p in metrics:
        
        if (len(probs) !=0) & (p != roc_auc_score) & (p != average_precision_score) & (p != metric_aupr_macro):
            y_pred_ = np.argmax(y_pred,axis=1)
            y_pred_ = np.array(pd.get_dummies(y_pred_))
        else:  
            y_pred_ = y_pred
        
        extremes = get_CI_boot(y_true,y_pred_,p,boot,metric_average) #atencao aqui ao boot
        
        e.append(extremes)
        if p == accuracy_score :
            myp.append((round(p(y_true,y_pred_), 2)))
            
        else:
            if (len(probs) !=0) & (p != roc_auc_score) & (p != average_precision_score) & (p != metric_aupr_macro):
                y_pred_ = np.argmax(y_pred,axis=1)
                y_pred_ = np.array(pd.get_dummies(y_pred_))
            else:
                y_pred_ = y_pred   
            
            if p == spec:

                specificity1 = spec(y_true[:,0],y_pred_[:,0])
                specificity2 = spec(y_true[:,1],y_pred_[:,1])
                specificity_ = (specificity1 + specificity2)/2
                
                myp.append((round(specificity_, 2)))
                
            elif p == metric_aupr_macro:
                precisions, recalls, _ = precision_recall_curve(y_true.ravel(), y_pred_.ravel()) 
                # precision1, recall1, _ = precision_recall_curve(real[:,0], sample[:,0]) 
                # precision2, recall2, _ = precision_recall_curve(real[:,1], sample[:,1]) 
                # precisions = (precision1 + precision2)/2
                # recalls = (recall1 + recall2)/2
                aucs = auc(recalls, precisions)    
                myp.append((round(aucs, 2)))
            else:
                myp.append((round(p(y_true,y_pred_,average=metric_average), 2)))
        mye.append(str(' ['+ str(round(extremes[0], 2))  +'-'+  str(round(extremes[1], 2)) +']'))
    
    df1 = np.transpose(pd.DataFrame(myp, index=['AUROC','ACC','Recall','Spec','F1','AP','PPV','AUPRC']))
    df2 = np.transpose(pd.DataFrame(mye, index=['AUROC','ACC','Recall','Spec','F1','AP','PPV','AUPRC']))
    boot_all_macro = pd.concat([df1,df2]) 
    
    myp = []
    mye = []
    
    metrics = [roc_auc_score, accuracy_score,  recall_score, specificity_score,
               f1_score, average_precision_score, precision_score, metric_aupr_micro]
    
    boot=1000
    metric_average = "micro"
       
    e = []
    for p in metrics:
        
        if (len(probs) !=0) & (p != roc_auc_score) & (p != average_precision_score) & (p != metric_aupr_micro):
            y_pred_ = np.argmax(y_pred,axis=1)
            y_pred_ = np.array(pd.get_dummies(y_pred_))
        else:  
            y_pred_ = y_pred
        
        extremes = get_CI_boot(y_true,y_pred_,p,boot,metric_average) #atencao aqui ao boot
        
        e.append(extremes)
        if p == accuracy_score :
            myp.append((round(p(y_true,y_pred_), 2)))
            
        else:
            if (len(probs) !=0) & (p != roc_auc_score) & (p != average_precision_score) & (p != metric_aupr_micro):
                y_pred_ = np.argmax(y_pred,axis=1)
                y_pred_ = np.array(pd.get_dummies(y_pred_))
            else:
                y_pred_ = y_pred   
            
            if p == specificity_score:
                myp.append((round(p(y_true[:,1],y_pred_[:,1],average=metric_average), 2)))
            elif p == metric_aupr_micro:
                precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_pred_.ravel())
                myp.append((round(auc(recall["micro"], precision["micro"]), 2)))
            else:
                myp.append((round(p(y_true,y_pred_,average=metric_average), 2)))
        mye.append(str(' ['+ str(round(extremes[0], 2))  +'-'+  str(round(extremes[1], 2)) +']'))
    
    df1 = np.transpose(pd.DataFrame(myp, index=['AUROC','ACC','Recall','Spec','F1','AP','PPV','AUPRC']))
    df2 = np.transpose(pd.DataFrame(mye, index=['AUROC','ACC','Recall','Spec','F1','AP','PPV','AUPRC']))
    boot_all_micro = pd.concat([df1,df2]) 
                        
    # myp = []
    # mye = []
    
    # metrics = [roc_auc_score, accuracy_score,  recall_score, specificity_score,
    #            f1_score, average_precision_score, precision_score]
    
    
       
    #%% Specificity and sensitivity (recall) ##################################
    
    def spec(y_true,y_pred):
        TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()
        return TN/(TN+FP)   
    
    def sens(y_true,y_pred):
        TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()
        return TP/(TP+FN) 
    
    def metric_aupr(y_true, y_pred):
        from sklearn.metrics import (make_scorer, roc_auc_score, cohen_kappa_score,confusion_matrix, precision_score,
                                 accuracy_score, auc, average_precision_score, f1_score,
                                 precision_recall_curve, recall_score, roc_curve)          

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auc = auc(recall, precision)
        return auc
    
    #%% Bootsrapping results in test per outcome ##################################
    
    if (len(probs) !=0):
         y_pred_ = y_pred
         y_pred = (y_pred0).astype(np.int)
         y_pred = np.array(pd.get_dummies(y_pred))
    
    df1 = []
    df2 = []
    
    boot=1000
    for outcome in range(0,len(labels)):
    
        y_pred_outcome = y_pred[:,outcome]
        
        y_true_outcome = y_true[:,outcome]
        
        myp = []
        mye = []
        
        metrics = [roc_auc_score, accuracy_score, sens, spec, f1_score, 
                   average_precision_score, precision_score, metric_aupr] 
        
        y_true_outcome = np.array(y_true_outcome)
        
        for p in metrics:
            if (len(probs) !=0) & ((p == roc_auc_score) | (p == average_precision_score) | (p == metric_aupr)):
                y_pred_outcome = y_pred_[:,outcome]
            else:
                y_pred_outcome =  y_pred[:,outcome]   
            extremes = get_CI_boot_outcome(y_true_outcome,y_pred_outcome,p,boot) #atencao aqui ao boot
            myp.append((round(p(y_true_outcome,y_pred_outcome), 2)))
            mye.append(str(' ['+ str(round(extremes[0], 2))  +'-'+  str(round(extremes[1], 2)) +']'))
        
        df1.append(myp)                                                   
        df2.append(mye)
    
    df1 = pd.DataFrame(df1, columns = ['AUROC','ACC','Recall','Spec','F1','AP','PPV','AUPRC'])
    df2 = pd.DataFrame(df2, columns = ['AUROC','ACC','Recall','Spec','F1','AP','PPV','AUPRC'])
         
    boot_label = pd.concat([df1,df2], axis = 1)
     
    return boot_all_micro, boot_all_macro, boot_label
        
        
