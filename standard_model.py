from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import os
import pandas as pd
import numpy as np


from tropical_layers import (TropicalMaxPool2D, TropicalReLU,
                             TropicalFlatten,# TropicalEmbed,
                             TropicalConv1D,TropicalMaxPool1D,
                             TropicalConv2D, TropicalDense,
                             tropical_objects,tropical_equivalents,
                             TropicalSequential,
                            )

'''
Standardization of models

For now, limit to keras models using
Dense, Conv2D, and Flatten layers
relu nonlinearities

Flatten must be proceeded by relu layer
Should be linear (not sigmoid terminated)
'''




def fashion_model(input_shape=(28,28)):
    model= keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    return model

    #Model2   #also works fine on mnist
def frog_ship_conv(input_shape):
    model =keras.Sequential([
    keras.layers.Conv2D(64,3,activation='relu',input_shape=input_shape),
    #keras.layers.MaxPool2D( (2,2) ),#recently commented
    ##keras.layers.Conv2D(128,3,2,activation='relu'),
    keras.layers.Conv2D(128,3,activation='relu'),
    #keras.layers.MaxPool2D( (2,2) ),#recently commented
    keras.layers.Conv2D(64,3,activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(64,activation='relu'), #-1 if commented out

    keras.layers.Dense(1), #XE from logits
    #keras.layers.Dense(1,activation='sigmoid'),
    ])
    return model

def mnist_conv1(input_shape):
    ksz=(3,3)#kernel size
    sds=(2,2)#strides
    return keras.Sequential([
        keras.layers.Conv2D(32,ksz,sds,activation='relu',name='L1',
                                 input_shape=input_shape),
        keras.layers.Conv2D(64,ksz,sds,activation='relu',name='L2'),
        keras.layers.Conv2D(64,ksz,sds,activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(1),

#        keras.layers.Conv2D(32,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.Flatten(),
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(1,activation='sigmoid'),
        ])

def first_trop(input_shape):
    ksz=(3,3)#kernel size
    sds=(2,2)#strides
    return TropicalSequential([
        TropicalConv2D(32,ksz,sds,name='L1',input_shape=input_shape),
        TropicalReLU(),
        TropicalConv2D(64,ksz,sds,name='L2'),#,activation='relu',name='L2'),
        TropicalReLU(),
        TropicalConv2D(64,ksz,sds),#,activation='relu'),
        TropicalReLU(),
        TropicalFlatten(),
        TropicalDense(64),#,activation='relu'),
        TropicalReLU(),
        TropicalDense(1),
        ])

def smallest_conv(input_shape):
    ksz=(3,3)#kernel size
    sds=(2,2)#strides
    return TropicalSequential([
        TropicalConv2D(8,ksz,sds,name='L1',input_shape=input_shape),
        TropicalReLU(),
        TropicalConv2D(8,ksz,sds,name='L2'),#,activation='relu',name='L2'),
        TropicalReLU(),
        TropicalConv2D(16,ksz,sds),#,activation='relu'),
        TropicalReLU(),
        TropicalFlatten(),
        TropicalDense(10),#,activation='relu'),
        TropicalReLU(),
        TropicalDense(1),
        ])

#--ecg
def small_conv1d(input_shape):# 0.69 or so acc
    ksz=4#kernel size
    sds=2#strides
    return keras.Sequential([
        keras.layers.Conv1D(32,ksz,sds,activation='relu',name='L1',
                                 input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=2, strides=1),
        keras.layers.Conv1D(64,ksz,sds,activation='relu',name='L2'),
        keras.layers.MaxPooling1D(pool_size=2, strides=1),
        keras.layers.Conv1D(64,ksz,sds,activation='relu',name='L3'),
        keras.layers.Flatten(),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(1),
        ])

#No good use TropicalReLU
#def trop_conv1d(input_shape):
#    ksz=4#kernel size
#    sds=2#strides
#    return TropicalSequential([
#        TropicalConv1D(32,ksz,sds,activation='relu',name='L1',
#                                 input_shape=input_shape),
#        TropicalMaxPool1D(pool_size=2, strides=1),
#        TropicalConv1D(64,ksz,sds,activation='relu',name='L2'),
#        TropicalMaxPool1D(pool_size=2, strides=1),
#        TropicalConv1D(64,ksz,sds,activation='relu',name='L3'),
#        TropicalFlatten(),
#        TropicalDense(64,activation='relu'),
#        TropicalDense(1),
#        ])


##Just like trop1d_small, but with reduced grid dim
def trop1d_encode4(input_shape):##about 72, 72.48
    ksz=4#kernel size
    sds=2#strides
    return keras.Sequential([
        TropicalConv1D(64,ksz+2,sds,input_shape=input_shape),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalConv1D(64,ksz+2,sds),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalConv1D(32,ksz,sds),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalConv1D(16,ksz,sds),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalFlatten(),
        TropicalDense(32),
        TropicalReLU(),
        TropicalDense(32),
        TropicalReLU(),

        TropicalDense(4),
        TropicalReLU(),

        TropicalDense(32),
        TropicalReLU(),
        TropicalDense(32),
        TropicalReLU(),
        TropicalDense(64),
        TropicalReLU(),
        TropicalDense(1),
        ])

#just like trop1d_encode4 but with grid dim 5
def trop1d_small(input_shape):##about 72, 72.48
    ksz=4#kernel size
    sds=2#strides
    return keras.Sequential([
        TropicalConv1D(64,ksz+2,sds,input_shape=input_shape),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalConv1D(64,ksz+2,sds),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalConv1D(32,ksz,sds),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalConv1D(16,ksz,sds),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalFlatten(),
        TropicalDense(32),
        TropicalReLU(),
        TropicalDense(32),
        TropicalReLU(),

        TropicalDense(5),
        TropicalReLU(),

        TropicalDense(32),
        TropicalReLU(),
        TropicalDense(32),
        TropicalReLU(),
        TropicalDense(64),
        TropicalReLU(),
        TropicalDense(1),
        ])


#decrease number of neurons in terminal layers compared t1d_small
def trop1d_tiny(input_shape):##about 72, 72.48
    ksz=4#kernel size
    sds=2#strides
    return keras.Sequential([
        TropicalConv1D(64,ksz+2,sds,input_shape=input_shape),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalConv1D(64,ksz+2,sds),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalConv1D(32,ksz,sds),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalConv1D(16,ksz,sds),
        TropicalReLU(),
        TropicalMaxPool1D(pool_size=2, strides=1),

        TropicalFlatten(),
        TropicalDense(32),
        TropicalReLU(),
        TropicalDense(32),
        TropicalReLU(),

        TropicalDense(5),
        TropicalReLU(),

        TropicalDense(5),
        TropicalReLU(),
        TropicalDense(5),
        TropicalReLU(),
        TropicalDense(10),
        TropicalReLU(),
        TropicalDense(1),
        ])



def conv1d_small(input_shape):##about 72
    ksz=4#kernel size
    sds=2#strides
    return keras.Sequential([
        keras.layers.Conv1D(64,ksz+2,sds,activation='relu',
                                 input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=2, strides=1),

        keras.layers.Conv1D(64,ksz+2,sds,activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2, strides=1),

        keras.layers.Conv1D(32,ksz,sds,activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2, strides=1),

        keras.layers.Conv1D(16,ksz,sds,activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2, strides=1),

        keras.layers.Flatten(),
        keras.layers.Dense(32,activation='relu'),
        keras.layers.Dense(32,activation='relu'),

        keras.layers.Dense(5,activation='relu'),

        keras.layers.Dense(32,activation='relu'),
        keras.layers.Dense(32,activation='relu'),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(1),
        ])

def conv1d_fc(input_shape):##about 75%acc 
    ksz=4#kernel size
    sds=2#strides
    return keras.Sequential([
        keras.layers.Conv1D(64,ksz+2,sds,activation='relu',name='C1',
                                 input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=2, strides=1),

        keras.layers.Conv1D(64,ksz+2,sds,activation='relu',name='C2',),
        keras.layers.MaxPooling1D(pool_size=2, strides=1),

        keras.layers.Conv1D(64,ksz,sds,activation='relu',name='C3'),
        keras.layers.MaxPooling1D(pool_size=2, strides=1),

        keras.layers.Flatten(),
        keras.layers.Dense(32,activation='relu',name='D1'),
        keras.layers.Dense(32,activation='relu',name='D2'),
        keras.layers.Dense(64,activation='relu',name='D3'),
        keras.layers.Dense(1,name='D4'),
        ])
#(0, ':', TensorShape([20, 150, 1]))
#(1, ':', TensorShape([20, 73, 64]))
#(2, ':', TensorShape([20, 72, 64]))
#(3, ':', TensorShape([20, 34, 64]))
#(4, ':', TensorShape([20, 33, 64]))
#(5, ':', TensorShape([20, 15, 32]))
#(6, ':', TensorShape([20, 14, 32]))
#(7, ':', TensorShape([20, 448]))
#(8, ':', TensorShape([20, 6]))
#(9, ':', TensorShape([20, 32]))
#(10, ':', TensorShape([20, 64]))


def deepecg_v1(input_shape):
    ksz=4#kernel size
    sds=2#strides
    pad='same'
    return keras.Sequential([
        #keras.layers.Conv1D(320,ksz,sds,activation='relu',name='L1',
        keras.layers.Conv1D(320,24,1,activation='relu',name='L1',
                                 input_shape=input_shape,padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2, strides=2,padding=pad),


        keras.layers.Conv1D(256,16,1,dilation_rate=2,activation='relu',name='L2',padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(256,16,1,dilation_rate=4,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(256,16,1,dilation_rate=4,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(256,16,1,dilation_rate=4,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),

        keras.layers.Conv1D(128,8,1,dilation_rate=4,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2, strides=2,padding=pad),

        keras.layers.Conv1D(128,8,1,dilation_rate=6,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(128,8,1,dilation_rate=6,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(128,8,1,dilation_rate=6,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(128,8,1,dilation_rate=6,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),

        keras.layers.Conv1D(128,8,1,dilation_rate=8,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2, strides=2,padding=pad),

        keras.layers.Conv1D(64,8,1,dilation_rate=8,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(64,8,1,dilation_rate=8,activation='relu',padding=pad),
        keras.layers.BatchNormalization(),

        #global ave pooling, skipped for a sec
        #keras.layers.GlobalAveragePooling1D()
        keras.layers.Flatten(),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(1),

        ])



#Noisy linear models (prev)
#Fig_Model_0505_005701_noise1D1A3#*
#Fig_Model_0505_003620_noise2D1A3#*
#Fig_Model_0505_012310_noise3D1A3

def inspect_activation_shapes(model,inputs):
    A=inputs
    for l,L in enumerate(model.layers):
        print(l,':',A.shape)
        A=L(A)
    print('out',':',A.shape)


if __name__=='__main__':
    ##debug stuff
    from standard_data import ecg_notnoisy,peak
    datasets,info=ecg_notnoisy()
    train_ds=datasets['train']
    bat=peak(train_ds.batch(20))
    X20=bat['x']

    trop =trop1d_small(info.input_shape)
    inspect_activation_shapes(trop, X20)


    #model=conv1d_fc(info.input_shape)
    #inspect_activation_shapes(model, X20)

