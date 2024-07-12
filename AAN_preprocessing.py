from sklearn.metrics import cohen_kappa_score,accuracy_score
import numpy as np


from scipy import signal
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import math

import numpy as np
from scipy.linalg import sqrtm
import scipy.io

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



def get_source_data(nSub):
    n_channels = 22
    n_tests = 6*48  
    window_Length = 7*250 
    # train data
    data_path = ''
    total_data = scipy.io.loadmat(data_path+'A0%dT.mat' % nSub)
    train_label = np.zeros(n_tests)
    train_data = np.zeros((n_tests, n_channels, window_Length))
    NO_valid_trial = 0
    a_data = total_data['data']
    all_trials = True
    for ii in range(0,a_data.size):
        
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
            if(a_artifacts[trial] != 0 and not all_trials):
                continue
            train_data[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
            # train_data[NO_valid_trial,:,:] /=np.max(np.abs(train_data[NO_valid_trial,:,:]))
            train_label[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial +=1

    train_data = train_data[:, :, int(1.5*250):6*250]
    
    test_tmp = scipy.io.loadmat(data_path+'A0%dE.mat' % nSub)
    test_label = np.zeros(n_tests)
    test_data = np.zeros((n_tests, n_channels, window_Length))
    NO_valid_trial = 0
    a_data = test_tmp['data']
    all_trials = True
    for ii in range(0,a_data.size):
       
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
            if(a_artifacts[trial] != 0 and not all_trials ):
                continue
            test_data[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
            # test_data[NO_valid_trial,:,:] /=np.max(np.abs(test_data[NO_valid_trial,:,:]))
            test_label[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial +=1

    test_data = test_data[:,:, int(1.5*250):6*250]
    return np.expand_dims(train_data,1), to_categorical(train_label-1), np.expand_dims(test_data,1), to_categorical(test_label-1)


def normal_data(data):
    for i in range(data.shape[0]):
        data[i,:] -= np.mean(data[i,:])
        # data[i,:] = (data[i,:]-np.mean(data[i,:]))/(np.max(data[i,:])-np.min(data[i,:]))
        data[i,:] /=np.max(np.abs(data[i,:]))
        # data[i,:] /=np.std(data[i,:])
    return data

def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

def preprocess_ea_sub(train_data,test_data): #（N,1,22,1125） 
    R_bar = np.zeros((test_data.shape[2], test_data.shape[2]))
    for i in range(test_data.shape[0]):
        R_bar += np.dot(test_data[i,0,:], test_data[i,0,:].T)
    R_bar_mean_test = R_bar / test_data.shape[0]

    R_bar = np.zeros((test_data.shape[2], test_data.shape[2]))
    for i in range(train_data.shape[0]):
        R_bar += np.dot(train_data[i,0,:], train_data[i,0,:].T)
    R_bar_mean_train = R_bar / train_data.shape[0]

    for i in range(train_data.shape[0]):
        train_data[i,0,:] = np.dot(np.linalg.inv(sqrtm(R_bar_mean_train)), train_data[i,0,:])
    for i in range(test_data.shape[0]):
        test_data[i,0,:] = np.dot(np.linalg.inv(sqrtm(R_bar_mean_test)), test_data[i,0,:])
        # test_data[i,0,:] = np.dot(np.linalg.inv(sqrtm(R_bar_mean_train)), test_data[i,0,:])
    return train_data,test_data


def preprocess_ea_loso(train_data,test_data): #（N,1,22,1125）
    for k in range(8):
        R_bar = np.zeros((train_data.shape[2], train_data.shape[2]))
        for i in range(288):
            R_bar += np.dot(train_data[k*288+i,0,:], train_data[k*288+i,0,:].T)
        R_bar_mean = R_bar /288
        # assert (R_bar_mean >= 0 ).all(), 'Before squr,all element must >=0'

        for i in range(288):
            train_data[k*288+i,0,:] = np.dot(np.linalg.inv(sqrtm(R_bar_mean)), train_data[k*288+i,0,:])

    R_bar = np.zeros((test_data.shape[2], test_data.shape[2]))
    for i in range(test_data.shape[0]):
        R_bar += np.dot(test_data[i,0,:], test_data[i,0,:].T)
    R_bar_mean = R_bar / 288
    for i in range(test_data.shape[0]):
        test_data[i,0,:] = np.dot(np.linalg.inv(sqrtm(R_bar_mean)), test_data[i,0,:])
    return train_data,test_data


def get_LOSOdata(sub):
   
    train_data = np.zeros((1,1,22,1125))
    train_label = np.zeros((1,4))    
    for i in range(1,10):
        if(i!=sub):
            print(i)
            X_train, y_train_onehot, X_test, y_test_onehot = get_source_data(i)
           
            train_data = np.concatenate((train_data,X_train,X_test),axis=0)
            train_label = np.concatenate((train_label,y_train_onehot,y_test_onehot),axis=0)
        if(i==sub):
            X_train, y_train_onehot, X_test, y_test_onehot = get_source_data(i)
            # X_train,  X_test = standardize_data2(X_train,X_test,22)
            # X_train,  X_test = preprocess_ea2(X_train,  X_test) 
            test_data = np.concatenate((X_train,X_test),axis=0)
            test_label = np.concatenate((y_train_onehot,y_test_onehot),axis=0)
    train_data = np.delete(train_data,0,0) #删除轴1的第一行
    train_label= np.delete(train_label,0,0)
    # train_data, test_data = standardize_data2(train_data,test_data,22)
    # train_data, test_data = preprocess_ea2(train_data,test_data) 
    shuffle_num = np.random.permutation(len(train_data))
    train_data = train_data[shuffle_num, :]
    train_label = train_label[shuffle_num,:]
    return train_data,train_label,test_data,test_label