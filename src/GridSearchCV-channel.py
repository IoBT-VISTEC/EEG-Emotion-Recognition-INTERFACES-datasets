#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, PredefinedSplit
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import math

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# settings
nsubjects = 43
samples_per_subj = nclips = 15
print('nsubjects:', nsubjects, 'samples_per_subj:', samples_per_subj)

# Set up possible values of parameters to optimize over
p_grid = {'kernel': ['poly', 'rbf', 'sigmoid'],
          "C": [1, 10, 100],
          "gamma": [.01, .1], 
          "degree": [3, 4, 5],
          "coef0": [0, .01, .1]
         }

df = pd.DataFrame(columns=['dataset_id', 'fold', 'train_acc', 'train_F1', 'test_acc', 'test_F1', 
                   'best_kernel', 'best_coef0', 'best_degree', 
                   'best_gamma', 'best_C', 
                   'class_ratio_train (0/1)', 'class_ratio_test (0/1)'])

# prepare X, y
X1 = np.load('../data/EEG/feature_extracted/EEG_ICA.npy')
X2 = np.load('../data/E4/feature_extracted/TEMP.npy')
X3 = np.load('../data/E4/feature_extracted/BVP.npy')
X4 = np.load('../data/E4/feature_extracted/EDA.npy')
EEG = X1
E4 = np.concatenate((X2,X3,X4), axis=1)

Fea = np.concatenate((EEG,E4), axis=1)
print (EEG.shape, E4.shape, Fea.shape)


def save_to_csv(df, fname):
    df.to_csv('./results_SVM/'+emotion+'_'+fname+'.csv', index=False)


#### LEAVE ONE CLIP OUT, USE SOME CHANNELS AND ALL FREQ

ChannelSelect = [[0,1,2] , [0,1,3] , [0,1,6], [0,1,7] ,[4,5,2], [4,5,3], [4,5,6], [4,5,7],[2,3,6,7]]

# prepare data for channel selection
DataX1 = EEG
DataX2 = E4
DataX1 =DataX1.reshape(43,15,-1)
DataX2 =DataX2.reshape(43,15,-1)
DataX1 = np.transpose(DataX1, (1, 0, 2))
DataX2 = np.transpose(DataX2, (1, 0, 2))

DataAllCh=[]
DataCh_E4=[]
for j in ChannelSelect:
    DataCh=np.zeros([15,43,4*len(j)])
    for i,x in enumerate (j):
        DataCh[:,:,i*4:i*4+4] = DataX1[:,:,4*x:4*x+4]
    DataAllCh.append(DataCh)
    DataCh_E4.append(np.concatenate((DataCh,DataX2), axis=2))
    print(np.concatenate((DataCh,DataX2), axis=2).shape)

print('DataAllCh =', (len(DataAllCh), len(DataAllCh[0]), len(DataAllCh[0][1]), ))
print('DataCh_E4 =', (len(DataCh_E4), len(DataCh_E4[0]), len(DataCh_E4[0][1]), ))

# DataAllCh: select some channels and concat without E4 features
# shape = (9, nclips, nsubjects, nfeatures)

# DataCh_E4: select some channels and concat with E4 features
# shape = (9, nclips, nsubjects, nfeatures)

for emotion in ['Arousal', 'Valence']:

    for f in ['EEG_only', 'EEG_E4']:
        
        # ** SELECT f = 'EEG_only', 'EEG_E4'
        if f == 'EEG_only':
            Data = DataAllCh
        elif f == 'EEG_E4':
            Data = DataCh_E4
            
        df = pd.DataFrame(columns=['dataset_id', 'fold', 'train_acc', 'train_F1', 'test_acc', 'test_F1', 
                           'best_kernel', 'best_coef0', 'best_degree', 
                           'best_gamma', 'best_C', 
                           'class_ratio_train (0/1)', 'class_ratio_test (0/1)'])
        
        # use label from kmeans #
        label = np.load('../data/score/label/kmeans.npy') 
        y = label

        if(emotion == 'Arousal'):
            y = [0 if(kk== 0 or kk==2) else 1 for kk in y]
        elif(emotion == 'Valence'):
            y = [0 if(kk== 0 or kk==1) else 1 for kk in y]
        y=np.asarray(y)

        # reshape from y shape = (nsubjects * samples_per_subj)
        y = y.reshape(nsubjects, samples_per_subj)

        # transpose to (clips, subjects)
        y = np.transpose(y)

        # rearrange clips
        # to 0,1,2,5,6,7,10,11,12,others
        y = np.concatenate([y[0:3], y[5:8], y[10:13], y[3:5], y[8:10], y[13:16]])

        for dataset_id, X in enumerate(Data):
            print('==== dataset:', dataset_id, '====')
            # y is same for every set, X is changed
            # shape X = (nclips, nsubjs, nfeatures)

            # rearrange clips to 0,1,2,5,6,7,10,11,12,others
            X = np.concatenate([X[0:3], X[5:8], X[10:13], X[3:5], X[8:10], X[13:16]])
            assert X.shape[0] == nclips == len(y)
            print(X.shape, y.shape)

            # ** The following code is as same as 2)
            folds = KFold(n_splits=nclips) #to leave test set out

            clip_id = 0
            for train_index, test_index in folds.split(X):
                clip_id += 1
                print('test index', test_index[0])

                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]
                n = len(X_train)
                print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


                X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], -1)
                y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1])

                X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1], -1)
                y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1])
                print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


                # norm by train set
                scaler = MinMaxScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                
                
                # upsampling class that has lower number
                tmp_shape = X_test.shape
                upsampling = False
                if upsampling:
                    if not 'up' in f:
                        f+='_up'
                    nc0 = len(X_test[y_test==0])
                    nc1 = len(X_test[y_test==1])
                    if nc0==0 or nc1 == 0:
                        df = df.append({'dataset_id': ChannelSelect[dataset_id], 'fold': test_index[0],
                                'train_acc': '%.2f' % (-1), 'train_F1': '%.2f' % (-1), 
                                'test_acc': '%.2f' % (-1), 'test_F1': '%.2f' % (-1), 
                                'best_kernel': ' ', 'best_coef0': ' ', 
                                'best_degree': ' ', 'best_gamma': ' ', 
                                'best_C': ' ', 
                                'class_ratio_train (0/1)': str(len(y_train[y_train==0]))+'/'+str(len(y_train[y_train==1])), 
                                'class_ratio_test (0/1)': str(len(y_test[y_test==0]))+'/'+str(len(y_test[y_test==1]))}, 
                               ignore_index=True)
                        save_to_csv(df, f)
                        continue

                    nc = [nc0, nc1]
                    higher = np.argmax(nc)
                    lower = abs(higher-1)

                    new_x = X_test
                    new_y = y_test
                    while True:
                        print('nc', nc)
                        if nc[lower] + len(new_y) < nc[higher]*2:
                            new_x = np.concatenate([new_x, X_test[y_test==lower]])
                            new_y = np.concatenate([new_y, y_test[y_test==lower]])
                        else:
                            remain = nc[higher]*2 - len(new_y)
                            new_x = np.concatenate([new_x, X_test[y_test==lower][0:remain]])
                            new_y = np.concatenate([new_y, y_test[y_test==lower][0:remain]])
                            break
                    X_test = np.array(new_x)
                    y_test = np.array(new_y)
                    print(X_test.shape, y_test.shape, len(y_test[y_test==0]), len(y_test[y_test==1]))
                    assert X_test.shape[1] == tmp_shape[1]
                    assert X_test.shape[0] >= tmp_shape[0]
                    assert len(y_test[y_test==0]) == len(y_test[y_test==1])
                

                # leave person out each fold
                test_fold = np.concatenate([[0]*43, [1]*43, [2]*43, [3]*43, [4]*43, 
                                            [5]*43, [6]*43, [7]*43, [-1]*((nsubjects*(nclips-1))-(8*nsubjects))])
                gkf = PredefinedSplit(test_fold)
                print('split train set into:', gkf.get_n_splits(), 'folds')

                # We will use a Support Vector Classifier with class_weight balanced
                svm = SVC(class_weight = 'balanced')
                clf_best = GridSearchCV(estimator=svm, 
                                   param_grid=p_grid, 
                                   cv=gkf, 
                                   iid=False, 
                                   scoring=['accuracy', 'balanced_accuracy', 'f1_macro'],
                                   refit = 'f1_macro') # get params that give best 'refit' value
                clf_best.fit(X_train, y_train)
                y_pred = clf_best.predict(X_train)
                train_f1 = clf_best.best_score_

                print('clf_best best_score:', train_f1)
                print('clf_best best_params_:', clf_best.best_params_)
                c0_train = len(y_train[(y_train==0) & (y_train==y_pred)])
                c1_train = len(y_train[(y_train==1) & (y_train==y_pred)])
                train_acc = (c0_train+c1_train)/len(y_train)
                print('clf_best train correct: c0_train =', c0_train, '/', len(y_train[y_train==0]), 
                      'clf_best c1_train =', c1_train, '/', len(y_train[y_train==1]), 'from', len(y_train), '=', train_acc)


                # We will use a Linear SVC which allows regularization
                # Set up possible values of parameters to optimize over
                p_grid2 = {"C": [1, 10, 100],
                           "penalty": ['l1', 'l2']
                          }

                # sklearn: prefer dual=False when n_samples > n_features
                linear_svm = LinearSVC(class_weight = 'balanced', dual=False) 
                clf_linear_best = GridSearchCV(estimator=linear_svm, 
                                   param_grid=p_grid2, 
                                   cv=gkf, 
                                   iid=False, 
                                   scoring=['accuracy', 'balanced_accuracy', 'f1_macro'],
                                   refit = 'f1_macro') # get params that give best 'refit' value
                clf_linear_best.fit(X_train, y_train)
                y_pred = clf_linear_best.predict(X_train)
                train_f1 = clf_linear_best.best_score_

                print('clf_linear_best best_score:', train_f1)
                print('clf_linear_best best_params_:', clf_linear_best.best_params_)
                c0_train = len(y_train[(y_train==0) & (y_train==y_pred)])
                c1_train = len(y_train[(y_train==1) & (y_train==y_pred)])
                train_acc = (c0_train+c1_train)/len(y_train)
                print('clf_linear_best train correct: c0_train =', c0_train, '/', len(y_train[y_train==0]), 
                      'clf_linear_best c1_train =', c1_train, '/', len(y_train[y_train==1]), 
                      'from', len(y_train), '=', train_acc)


                if clf_best.best_score_ > clf_linear_best.best_score_:
                    clf = clf_best
                    print('using SVC')
                    degree = clf.best_params_['degree']
                    gamma = clf.best_params_['gamma']
                    kernel = clf.best_params_['kernel']
                    penalty = '-'
                    coef0 = clf.best_params_['coef0']
                else:
                    clf = clf_linear_best
                    print('using LinearSVC')
                    degree = '-'
                    gamma = '-'
                    kernel = 'Linear'
                    penalty = clf.best_params_['penalty']
                    coef0 = '-'

                y_test_pred = clf.predict(X_test)
                c0_test = len(y_test[(y_test==0) & (y_test==y_test_pred)])
                c1_test = len(y_test[(y_test==1) & (y_test==y_test_pred)])
                test_acc = (c0_test+c1_test)/len(y_test)
                test_f1 = f1_score(y_test, y_test_pred, average='macro')
                print('test correct: c0_test =', c0_test, '/', len(y_test[y_test==0]), 
                      'c1_test =', c1_test, '/', len(y_test[y_test==1]), 'from', len(y_test), '=', test_acc)
                print()
                df = df.append({'dataset_id': ChannelSelect[dataset_id], 'fold': test_index[0],
                                'train_acc': '%.2f' % (train_acc*100), 'train_F1': '%.2f' % (train_f1*100), 
                                'test_acc': '%.2f' % (test_acc*100), 'test_F1': '%.2f' % (test_f1*100), 
                                'best_kernel': kernel, 'best_coef0': coef0, 
                                'best_degree': degree, 'best_gamma': gamma, 
                                'best_C': clf.best_params_['C'], 'penalty': penalty,
                                'class_ratio_train (0/1)': str(len(y_train[y_train==0]))+'/'+str(len(y_train[y_train==1])), 
                                'class_ratio_test (0/1)': str(len(y_test[y_test==0]))+'/'+str(len(y_test[y_test==1]))}, 
                               ignore_index=True)

                save_to_csv(df, f+'_chSelect')

                # leave one clip out to be test set (only clip 0-8)
                if clip_id == 9:
                    break
                    
        del Data, X_train, y_train, X_test, y_test, X, y



