from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import os
import importlib
import pandas as pd
import numpy as np

from config import get_config
from utils import prepare_dirs_and_logger,save_config,make_folders
from utils import relu,fg_diff,hinge
from utils import batch_inner,batch_outer
from utils import outer_inner,cwise_inner
from utils import SH,DIFF,EQUAL #db handy
from utils import logit_bxe #cross entropy helper
from utils import f_minus_g

import standard_data
import standard_model
from standard_data import peak,inspect_dataset,tuple_splits
from standard_model import tropical_objects #help load custom layers

from ArrayDict import ArrayDict
import time##db

from oldstyle.vis_utils import (split_posneg , get_path, get_np_network,
                        get_neuron_values, splitL, load_weights,
                        resample_grid,vec_get_neuron_values,
                        get_del_weights, subplots)
from tboard import file2number



import glob2
from vis_utils import (sample_grid,subplots,paint_data,
                       paint_binary_contourf,splitL,
                       draw_binary_contour,surf )
from config import load_config

'''
Trying to explore why greedy approach to 2d models is failing
maybe also doing some visualization along the way
'''


data1_models=['./logs/v1FigModel_1120_195114_data1',
              './logs/v1FigModel_1119_061719_noise1data1',
              './logs/v1FigModel_1120_193930_noise2data1',
              './logs/v1FigModel_1120_194302_noise3data1']

if __name__=='__main__':
    plt.close('all')

    #load_path=data1_models[0]#no noise
    load_path=data1_models[1]
    #load_path=data1_models[2]
    #load_path=data1_models[3]
    config=load_config(load_path)
    gX=sample_grid()

    record_dir=os.path.join(load_path,'records')
    id_str=str(file2number(load_path))
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

#-------------Begin Data--------------#
    #datasets,info=standard_data.v1_noisy1_data1()
    datasets,info=getattr(standard_data,config['data'])()

    ds_train=datasets['train']
    ds_input=ds_train.map(lambda e:e['x']).batch(1)

    train_size=info.train_size
    test_size=info.test_size

    if train_size<1000:
        np_data=peak(ds_train.batch(1000))
        tfX=np_data['x']
        tfY=np_data['y']
        npX,npY=tfX.numpy(),tfY.numpy()

    #---reference batch for datasets---#
    db_bat=peak(ds_train.batch(50))
    X=db_bat['x']
    Y=db_bat['y']
    X20=X[:20]

    #lookupX=np.concatenate([tf.expand_dims(db_bat['idx'],-1),X],-1)
    lookupX=np.concatenate([np.stack([db_bat['idx'],Y],-1),X],-1)#idx,Y,X1,X2


#-------------Begin Model--------------#

    m_files=glob2.glob(load_path+'/checkp*/*')
    if len(m_files)!=1:
        raise ValueError('Expected 1 model file but got ',m_files)
    model_file=m_files[0]
    model=keras.models.load_model(model_file,#pass h5 file
        custom_objects=tropical_objects)



# -----   new  -----  #
    def lyr2state(layer):
        return tf.cast( tf.greater(layer.output, 0.), tf.float32 )
    kernel_layers=filter(lambda L:hasattr(L,'kernel'),model.layers)
    state_list=[lyr2state(L) for L in kernel_layers]
    State=keras.Model(inputs=model.inputs,outputs=state_list)

    sigl=State(X)
    Sig=np.concatenate(sigl,axis=-1)#no conv


    SigH=np.concatenate(sigl[:-1],axis=-1)#no output
    unqSig=np.unique(SigH,axis=0)

    #For ref to calc Sig0
        #IdxPlus =Inv[np.where(fl_paths_pred==1)[0]]
        #IdxMinus=Inv[np.where(fl_paths_pred==0)[0]]
        #Idx0=np.intersect1d(IdxPlus,IdxMinus)



    ##########################
    #analyze sig behavior on samples
    LHL=model.layers[-1]
    sigHL=sigl[-2].numpy()
    kerHL=LHL.kernel.numpy()
    LHL=model.layers[-1]
    reorder=LHL.kernel.numpy().argsort(axis=0).flatten()
    sigHL=sigHL[:,reorder]
    kerHL=kerHL[reorder]

    std_sigHL=np.std(sigHL,axis=0)
    nontrivial=std_sigHL.nonzero()[0]
    sigHL=sigHL[:,nontrivial]#toss neurons that never change
    kerHL=kerHL[nontrivial]

    firstpos=np.where(kerHL>=0.)[0].min()
    tau_unique,tau_inv,tau_cnts=np.unique(sigHL[:,:firstpos],
                axis=0,return_inverse=True,return_counts=True)


    mu_unique,mu_inv,mu_cnts=np.unique(sigHL[:,firstpos:],
                axis=0,return_inverse=True,return_counts=True)


    g_sigl=State(gX)
    gY=model(gX)
    g_Sig=np.concatenate(g_sigl,axis=-1)
    neuron_states=splitL(g_Sig)

    n0=neuron_states[0]

    fig,ax=subplots()
    paint_data(ax,npX,npY)
    paint_binary_contourf(ax,gX,gY,0.)
    fname=os.path.join(record_dir,id_str+'_net_classif.pdf')
    plt.savefig(fname)

    fig,ax=subplots()
    paint_data(ax,npX,npY)
    paint_binary_contourf(ax,gX,gY,0.)
    fname=os.path.join(record_dir,id_str+'_statebdries.pdf')
    for neuron in neuron_states:
        draw_binary_contour(ax,gX,neuron)
    plt.savefig(fname)


    surf(gX,gY)
    fname=os.path.join(record_dir,id_str+'_net_surf.pdf')
    plt.savefig(fname)

        ##messy##



    print('WARN: plotting function is off. call plt.show')
    #plt.show(block=False)
    #plt.show()








