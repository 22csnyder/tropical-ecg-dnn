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

from trop_ops import *


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





    #Xinfo=np.concatenate([X,X[:,:1]-X[:,1:],model(X)],axis=-1)
    Xinfo=np.concatenate([np.arange(len(X)).reshape([-1,1]),X,X[:,:1]-X[:,1:],model(X),tf.reshape(Y,[-1,1])],axis=-1)


    inspect=np.array([3,16,26,27,31,46,43,37,22,36,20,47])
    x,y=np.array(X)[inspect],np.array(Y)[inspect]
    cidx=np.arange(len(inspect)).reshape([-1,1])#col of idx
    neg_idx=np.where(y==0)[0]
    neg_x=x[neg_idx]

    fg_x,fg_Linear,fg_Bias = model.fg_call( x )

    fgx      =tf.concat(fg_x,1)    #bhsz ,2
    kernel   =tf.stack(fg_Linear,1)#bhsz , 2,  x.shape
    bias     =tf.concat(fg_Bias,1) #bhsz , 2

    table= outer_inner( x, kernel )+bias #bsz x bsz(filters) x 2

    all_F=np.array(table[:,:,0])
    all_G=np.array(table[:,:,1])

    #neg_F=F[neg_idx][:,neg_idx]
    #neg_G=G[neg_idx][:,neg_idx]
    neg_F=all_F[neg_idx[:,np.newaxis],neg_idx]
    neg_G=all_G[neg_idx[:,np.newaxis],neg_idx]



    ##Finally this strategy works::
    neg_Fpme,neg_Gpme,neg_Hpme = row_reduce(neg_F,neg_G,True)
    neg_dF,neg_dG = map(row_normalize, [neg_F,neg_G])#another way
    neg_dFpme=-relu(neg_dG-neg_dF)# Add F to get Fpme
    neg_dGpme=-relu(neg_dF-neg_dG)# Add G to get Gpme

    #all_Fpme,all_Gpme,all_Hpme = row_reduce(all_F,all_G,True)
    all_dF,all_dG = map(row_normalize, [all_F,all_G])#another way

    np_relu=lambda x : np.maximum(x,0.*x)
    all_dFpme=-np_relu(all_dG-all_dF)# Add F to get Fpme
    all_dGpme=-np_relu(all_dF-all_dG)# Add G to get Gpme

    #feb2020--

    DF=all_dFpme
    DG=all_dGpme


    #Want blocks with Max or Min over subsets of columns to be zero
    #Investigate if Equiv to Taking max over columns, and min over ROWS (same block)
    DL = DF-DG


    #seems promising
    #row_normalize(neg_F-np.diag(neg_G))


    #dFpme+F = Fpme
    #dGpme+G = Gpme


    #
#    F,G=neg_F,neg_G
#    normF,normG=row_normalize(F),row_normalize(G)
#    dF,dG=normF,normG
#    #
#    F2= - F.T
#    G2= - G.T
#
#    H = maxproj(G2, F2).T
#    normH=maxproj(-normG.T,-normF.T).T
    #newF,newG=F+H,G+H



    ##stuff
#    dFnew = dF-maxproj(-dG.T,-dF.T).T  #seems to work?
#    dGnew = dG-maxproj(-dF.T,-dG.T).T  #not quite

    #Can make work for each individual row of f by rescaling rows of g to all
    #be zero in the same coordinate, then drop the zero column from rowF and G


    #  f4 - maxproj( -dG4.T , -f4.T).T



