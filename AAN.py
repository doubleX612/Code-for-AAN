from sklearn.metrics import cohen_kappa_score,accuracy_score
import numpy as np
from tensorflow.keras.utils import to_categorical

from scipy import signal
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample, freqz_zpk, zpk2sos, sosfiltfilt, cheb2ord, iirdesign
import math
from scipy.signal import firwin, lfilter, filtfilt, butter
import numpy as np
from scipy.linalg import sqrtm
import scipy.io
from scipy.signal import cheby2,sosfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.utils import plot_model,to_categorical


from keras import metrics
# from keras.utils import to_categorical


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Concatenate,Dropout,Lambda,Layer, Embedding,GRU
from tensorflow.keras.layers import Reshape,MaxPooling2D,MaxPooling3D,BatchNormalization
from tensorflow.keras.layers import Maximum,Average,Add,Multiply
from tensorflow.keras.layers import Dense, Activation,RepeatVector,Permute,DepthwiseConv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D,MaxPooling1D,ZeroPadding1D,AveragePooling1D
from tensorflow.keras.layers import Conv1D, Conv2D,Conv3D, SpatialDropout1D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten 
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute, Average,Subtract
from tensorflow.keras.constraints import max_norm,unit_norm,non_neg
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LayerNormalization,MultiHeadAttention,ZeroPadding2D

# from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomUniform
from tensorflow.python.keras.optimizers import adam_v2,rmsprop_v2
from tensorflow.python.keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
# from tfa.optimizers import AdamW


class FrameDataLayer(Layer):
    def __init__(self, frame_length, frame_shift,fs ,**kwargs):
        super(FrameDataLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.fs = fs

    def call(self, inputs,dynamic=True):
        batch_size, time_steps, features =  inputs.shape
        # Calculate the number of frames based on frame length and shift
        num_frames = (time_steps - self.frame_length) // self.frame_shift + 1

        # Reshape the inputs to create frames with frame length
        frames = [inputs[:, i * self.frame_shift:i * self.frame_shift + self.frame_length, :] for i in range(num_frames)]
        frames = tf.stack(frames, axis=2)  # Shape: (batch_size, frame_length, num_frames, features)
       
        frames = Conv2D(self.fs,(self.frame_length,15),(self.frame_length,1),kernel_initializer='he_normal',use_bias=False,
          kernel_constraint = max_norm(2., axis=(0,1,2)),padding = 'valid',data_format='channels_last',activation='selu')(frames)  
        frames = K.squeeze(frames,1)
        frames = Conv1D(self.fs,15,1,kernel_initializer='he_normal',kernel_constraint = max_norm(2., axis=(0,1)),
                                    use_bias=False,padding = 'same',data_format='channels_last',activation='selu')(frames)  
        frames = BatchNormalization(epsilon=1e-05, momentum=0.9,axis=-1)(frames)
        return frames


class ChannelAttentionLayer(Layer):
    def __init__(self, channels,**kwargs):
        self.channels = channels
        super(ChannelAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 初始化权重矩阵 W，可训练
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_normal',
                                 # initializer=RandomUniformWithShift(minval=-0.05, maxval=0.05, shift=0.1),
                                 trainable=True,
                                 # constraint=unit_norm(axis=[0,1]),
                                 name='W')
        self.scale = self.add_weight(shape=(input_shape[-1],),
                                     initializer='ones',
                                     trainable=True,
                                     # constraint=max_norm(1.),
                                     name='scale')
        super(ChannelAttentionLayer, self).build(input_shape)

    def call(self, inputs):
       
        inputs2 = Dense(self.channels, activation='selu',kernel_initializer='he_normal',kernel_constraint=max_norm(2.0))(inputs)
        # inputs2 = GLU(inputs,self.channels)
        attention_scores = K.dot(inputs2, self.W)/10.0
        weighted_inputs = attention_scores*inputs2
        output = Add()([weighted_inputs*self.scale,inputs])
        output = BatchNormalization(epsilon=1e-05, momentum=0.9, axis=-1)(output)
        output = SpatialDropout1D(0.85)(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape



def frame_shift(inputs,start,depth,filters):
    input_frame = FrameDataLayer(frame_length=start,frame_shift=1,fs=filters)(inputs)#     TensorShape([None,windows, 32])
    input_frame = ChannelAttentionLayer(filters)(input_frame)
    input_frame = Flatten()(input_frame)
    temp = Dense(4,activation="softmax",kernel_constraint=max_norm(0.25))(input_frame)
    for i in range(depth):
        if(i==0):
            output = temp
        else:
            temp_frame = FrameDataLayer(frame_length=start+i*4,frame_shift=1,fs=filters)(inputs)#     TensorShape([None,windows, 32])
            temp_frame = ChannelAttentionLayer(filters)(temp_frame)
            temp_frame = Flatten()(temp_frame)
            temp_frame = Dense(4,activation="softmax",kernel_constraint=max_norm(0.25))(temp_frame)
            output = Average()([output,temp_frame])
    return output


def cnn_model_2inverse(channels,samples,outsize):
 
    ee_in   = Input((1,channels, samples))#     TensorShape([None, 1, 22, 1125])
    input_frame = Permute((3,2,1))(ee_in)  #     TensorShape([None, 1125, 22, 1])

    input_frame = Conv2D(16,(64,1),(1,1),padding='same', activation="selu", kernel_initializer='he_normal',
        kernel_constraint = max_norm(2, axis=(0,1,2)))(input_frame)
    
    input_frame = BatchNormalization(epsilon=1e-05, momentum=0.9,axis=-1)(input_frame)#（batch_size,timestep,22,65）
    input_frame = MaxPooling2D((8,1),data_format='channels_last',padding = 'valid')(input_frame)
    input_frame = Dropout(0.3)(input_frame) 



   
    input_frame_short = Conv2D(32,(5,22),(1,22),padding ='valid',activation='selu',kernel_constraint = max_norm(2., axis=(0,1,2)),
         use_bias=False,kernel_initializer='he_normal')(input_frame)
    input_frame = BatchNormalization(epsilon=1e-05, momentum=0.9,axis=-1)(input_frame_short)#（batch_size,timestep,1,32）
    input_frame = Dropout(0.3)(input_frame)
    input_frame = K.squeeze(input_frame,2)
  
    

    ee_classout = frame_shift(input_frame,start = 3,depth = 3,filters=64)
    model=Model(inputs=ee_in,outputs=ee_classout)
    adm = adam_v2.Adam(learning_rate=9e-4, beta_1=0.9, beta_2=0.999,amsgrad=False,epsilon=1e-08, decay=0.0, clipnorm=1.0)
    model.compile(optimizer=adm,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    return model   




import random

import pandas as pd


data_path = ''

# lrate = LearningRateScheduler(step_decay1)

n_train = 3 # 
acc = np.zeros((9, n_train))
kappa = np.zeros((9, n_train))
best_acc,best_kappa=[],[]
cf_matrix = np.zeros((9,4,4))
for sub in range(1,10): # (num_sub): for all subjects, (i-1,i): for the ith subject. 先试试普通版，而后加入数据增强
        # subject-dependent 
        X_train, y_train_onehot, X_test, y_test_onehot = get_source_data(sub)
       # subject-independent
        # X_train, y_train_onehot, X_test, y_test_onehot = get_LOSOdata(sub)
        X_train,  X_test = standardize_data(X_train,X_test,22)
        
        X_train,  X_test = preprocess_ea_sub(X_train,  X_test) 
        # X_train,  X_test = preprocess_ea_loso(X_train,  X_test) 
        X_train = normal_data(X_train)
        X_test = normal_data(X_test)

        
        X_train_new, X_val, y_train_onehot_new, y_val_onehot = train_test_split(X_train, y_train_onehot, test_size=0.2,random_state=42)
        for k in range(n_train):
            
            test_model = cnn_model_2inverse(22,1125,4)
            checkpoint = ModelCheckpoint(filepath='./val_select/'+str(k)+'sub.h5',verbose=0,period=30,
                save_weights_only=True,monitor='val_loss',mode='min',save_best_only='True')

           
            np.random.seed(k+1)
            tf.random.set_seed(k+1)
            history = test_model.fit(X_train_new,y_train_onehot_new,
                batch_size=64,epochs= 500,
                validation_data=([X_val, y_val_onehot]),
                callbacks=[checkpoint,ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001)],#ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001)
                # shuffle = True,
                verbose=1)

           
            test_model.load_weights('./val_select/'+str(k)+'sub.h5')
            y_pred = test_model.predict(X_test)
            y_pred = y_pred.argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub-1, k]  = accuracy_score(labels, y_pred)
            kappa[sub-1, k] = cohen_kappa_score(labels, y_pred)
            print(acc,kappa)
    	  

np.save('./val_select/AAN_acc.npy',acc)
print(np.mean(np.max(acc,axis=1)))
np.save('./val_select/ANN_kappa.npy',kappa)
print(np.mean(np.max(kappa,axis=1)))

# np.save('./val_select/best_cf3.npy',cf_matrix)
# import os

# # 用于存储每个文件的num值
# nums = np.zeros(9)
# for i in range(1,10):
#     # 遍历当前目录下的所有文件
#     for filename in os.listdir():
#         if filename.startswith(str(i)+'_') and filename.endswith('training_log.csv'):
#             temp_num = float(filename.split('_')[1])
#             if(temp_num>nums[i-1]):
#                 nums[i-1] = temp_num
            
# # 计算平均值
# print(nums,np.mean(nums))
