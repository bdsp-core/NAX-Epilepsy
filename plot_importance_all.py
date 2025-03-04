import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

#------------------------------------------------------------------------
# Feature importance estimates - plot
#------------------------------------------------------------------------

def plt_importance_all(random_search, class_labels, matrix_features, features, N, model):

    
    
    if N == []:
        N = len(matrix_features.columns)  # total number of features
    

    def plot_topN(random_search, class_labels,N,a, model):
        
        clf = random_search #.best_estimator_
        
        if model == 'LR':
            coef0 = clf.steps[1][1].coef_
        elif ((model == 'RF') | (model == 'XGB')) & (len(class_labels) <= 2):
            coef0 = clf.steps[1][1].feature_importances_.reshape((1,len(a)))
        elif ((model == 'RF') | (model == 'XGB')) & (len(class_labels) == 3):
            coef01 = clf.steps[1][1].estimators_[0].feature_importances_.reshape((1,len(a)))
            coef02 = clf.steps[1][1].estimators_[1].feature_importances_.reshape((1,len(a)))
            coef03 = clf.steps[1][1].estimators_[2].feature_importances_.reshape((1,len(a)))
            coef0 = pd.concat([pd.DataFrame(coef01), pd.DataFrame(coef02)], axis = 0)
            coef0 = pd.concat([coef0, pd.DataFrame(coef03)], axis = 0).reset_index().drop(columns='index')
            coef0 = coef0.astype(float)
            coef0 = np.array(coef0)
        coef = abs(coef0)
        
        if model == 'LR':
            c = pd.DataFrame(clf.steps[1][1].coef_, columns = a)
        elif ((model == 'RF') | (model == 'XGB')) & (len(class_labels) <= 2):
            c = pd.DataFrame(clf.steps[1][1].feature_importances_.reshape((1,len(a))), columns = a)    
        elif ((model == 'RF') | (model == 'XGB')) & (len(class_labels) == 3):
            coef01 = clf.steps[1][1].estimators_[0].feature_importances_.reshape((1,len(a)))
            coef02 = clf.steps[1][1].estimators_[1].feature_importances_.reshape((1,len(a)))
            coef03 = clf.steps[1][1].estimators_[2].feature_importances_.reshape((1,len(a)))
            c = pd.concat([pd.DataFrame(coef01), pd.DataFrame(coef02)], axis = 0)
            c = pd.concat([c, pd.DataFrame(coef03)], axis = 0).reset_index().drop(columns='index')
            c = pd.DataFrame(np.array(c),  columns = a)
            c = c.astype(float)
        ind = (c == 0).all()
        ind = pd.DataFrame(ind,columns=['Bool'])
        a = ind.loc[(ind.Bool==False)].index   

        coef0 = coef0[:, (coef0 != 0).any(axis=0)]
        coef = abs(coef0)
        
        for i, class_label in enumerate(class_labels):
            print(i,class_label)
            feature_importance = coef[class_label]
            feature_signal = coef0[class_label]
            sorted_idx = np.argsort(np.transpose(feature_importance))
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            topN = sorted_idx[-N:]
            cols_names = []
            
            x1 = pd.DataFrame(feature_signal[topN])
            get_indexes_neg = x1.apply(lambda x: x[x<0].index)
            
            for j in topN:
                cols_names.append(a[j])         
            pos = np.arange(topN.shape[0]) + .5
            featfig = plt.figure(figsize=(12, 18))
            featax = featfig.add_subplot(1, 1, 1)
            b = featax.barh(pos, feature_importance[topN], align='center', color = '#0076c0', height=0.6)
            
            for ind in range(len(get_indexes_neg)):
                b[get_indexes_neg[0][ind]].set_color('#a30234')
            featax.set_yticks(pos)
            featax.set_yticklabels(np.array(cols_names), fontsize=30)
            featax.set_xlabel('Relative Feature Importance (%)', fontsize=30)
            
            plt.rcParams.update({'font.size': 30})
            
            if (i == 0) | ((i == len(class_labels)-1) & (len(class_labels)>5)):
                from matplotlib.lines import Line2D
                if len(get_indexes_neg) > 0:
                    custom_lines = [Line2D([0], [0], color='#0076c0', lw=4),
                                    Line2D([0], [0], color='#a30234', lw=4)]
                    featax.legend(custom_lines, ['Positive coefficient', 'Negative coefficient'], loc='lower right', prop={'size': 30})
                else:
                    custom_lines = [Line2D([0], [0], color='#0076c0', lw=4)]
                    featax.legend(custom_lines, ['Positive coefficient'], loc='lower right', prop={'size': 25})
   
            plt.rcParams.update({'font.sans-serif':'Calibri'})
        
    plot_topN(random_search,class_labels,N,features,model) # N = 50 first 50 tokens
    
   
            
    # get_indexes_neg = x2.apply(lambda x: x[x<0].index)
     
    # get_indexes_pos = x2.apply(lambda x: x[x>0].index)
       
    # mask_neg = x.iloc[get_indexes_neg[0]]
    
    # mask_pos = x.iloc[get_indexes_pos[0]]
