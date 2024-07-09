from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#from keras import backend as K
#from keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import os
import pandas as pd
import numpy as np
from utils import prepare_dirs_and_logger,save_config,make_folders
from utils import SH,DIFF,EQUAL #db handy
from utils import relu,batch_inner,batch_outer
from config import get_config

import standard_data
import standard_model

from standard_data import *
#from standard_data import peak,inspect_dataset,tuple_splits
from utils import logit_bxe

from v1_tropical_layers import (TropicalMaxPool2D, TropicalReLU,
                             TropicalFlatten, TropicalEmbed,
                             TropicalConv2D, TropicalDense,
                             tropical_objects,tropical_equivalents,
                            )
from ArrayDict import ArrayDict
import time##db

#?tf.function?
def local_linear(func,x):#local linear rep of func at x
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        outputs=func(x)
        if not isinstance(outputs,list):
            outputs=[outputs]

    Linear=[tape.gradient(y,x) for y in outputs]#each of shape x
    del tape

    linearX=[batch_inner(x,lin) for lin in Linear]
    Bias=[y-linX for y,linX in zip(outputs,linearX)]

    if len(outputs)==1:
        return Linear[0],Bias[0]#delist
    else:
        #return Linear,Bias #pair of lists
        return zip(Linear,Bias) #list of pairs

#?tf.function?
def eval_f_and_Df(func,x):
    return [ func(x),local_linear(func,x) ]


def trop2dnn(input_shape,trop_layers):
    '''Trys to return the equivalent
    relu dnn model given input_shape
    and tropical trop_layers.
    '''
    if input_shape[0] is None:
        input_shape=input_shape[1:]

    def has_equiv(layer):
        return isinstance( layer, tuple(tropical_equivalents.values()) )

    #Rid of Inputlayer, trivial embed layer#
    trop_layers=filter( has_equiv, trop_layers )

    inputNode=layers.Input(input_shape)
    equiv_layers=[L.equiv_layer for L in trop_layers]

    net_act=inputNode
    for lyr in equiv_layers:
        net_act=lyr(net_act)
    dnn_model=keras.Model(inputs=inputNode,outputs=net_act)

    return dnn_model


def Poly(fg):
    #can be streamlined with dfdg_Polys_x .w/.b

    if fg=='f':
        #wkey,bkey='F.w','F.b'
        #wkey,bkey='f.w','f.b'
        wkey,bkey='df.w','df.b'
    elif fg=='g':
        #wkey,bkey='G.w','G.b'
        #wkey,bkey='g.w','g.b'
        wkey,bkey='dg.w','dg.b'

    def trop_poly(x,lin_fns):
        kernel=lin_fns[wkey]
        bias  =lin_fns[bkey]

        lin_x = batch_outer( x, kernel ) + bias #(bsz_x, bsz_Lin, 1)
        poly_x= tf.reduce_max( lin_x , axis=1 )
        return poly_x

    return trop_poly


def calc_greedy_loss(batch_info,greedy_model_info):
    bh=batch_info
    ad_Greed=greedy_model_info

    x,y=bh['x'],bh['y']

    ###  Loss ###
    #y_01  = tf.cast(bh['y'],tf.int32)#was uint8#need int32/64 to indextrick
    #y_pm1 = 2*y_01 - 1
    y_hot = tf.one_hot(y,depth=2)#(bsz,2)#  has a 1 in first col if y=0
    F_poly_x=Poly('f')(x,ad_Greed)#(bsz_x,1)
    G_poly_x=Poly('g')(x,ad_Greed)#(bsz_x,1)
    FG_polys_x=tf.stack([F_poly_x,G_poly_x],axis=1)#(bsz_x,2,1)

    #fg_polys_x=tf.stack(fg_x,axis=1)#f,g at sig(x) at x
    fg_polys_x=bh['fg_polys_x']

    # net(x)        = py_poly_x - my_poly_x
    # greedy_approx = py_Poly_x - my_Poly_x
    #notice caps  all shapes are #(bsz,1)
    py_Poly_x=tf.boolean_mask(FG_polys_x,1-y_hot)#want y=1 -> get first col
    my_Poly_x=tf.boolean_mask(FG_polys_x,y_hot)  #every row is sampled once
    my_poly_x=tf.boolean_mask(fg_polys_x,y_hot)  #want y=1 -> return g_x
    want_positive = py_Poly_x - my_poly_x #how new is this sample?
    need_positive = py_Poly_x - my_Poly_x #are greedy_model.preds correct?

    hinge=lambda y_wx: relu(1 - y_wx)
    greedy_loss = hinge(want_positive)#use for determine next pt
    actual_loss = -need_positive#by how much are current predictions off?

    bh['actual loss']=actual_loss
    bh['greedy loss']=greedy_loss

    hinge=lambda y_wx: relu(1 - y_wx)
    loss_x=hinge( want_positive )
    return loss_x
    ###  Loss ###


if __name__=='__main__':
    print('db_greedysig.py')

    #look into later:
    #tf.keras.backend.clear_session()
    #keras.backend.set_floatx('float64')

    ##load demo data and model##


    datasets,info=mnist()
    #datasets,info=fashion_mnist()
    #datasets,info=cifar10()
    train_data,test_data=tuple_splits(datasets)
    info_shape=info.features['image'].shape
    iis=info.input_shape
    input_shape=tf.TensorShape(iis)
    db_bat=peak(datasets['train'].batch(20))
    X=db_bat['x']
    Y=db_bat['y']# shape (bsz,)
    O=tf.zeros_like(X)#looks like 0. is an Oh.
    l=tf.ones_like(X)#kinda looks like a 1
    db_bat25=peak(datasets['train'].batch(25))
    X25=db_bat25['x']
    X20=X
    ds_train=datasets['train']
    ds_input=ds_train.map(lambda e:e['x']).batch(1)
    input_shape=db_bat['x'].shape[1:]#equiv to tf.TS(info.is)

    ###prep training###
    train_size=info.splits['train'].num_examples
    test_size=info.splits['test'].num_examples
    buf=np.int(train_size*0.1)

    #Yes. was causing stall.hm
    #train_data=train_data.shuffle(buf)#was this causing stall?

    train_data=train_data.batch(32).repeat()
    test_data=test_data.batch(32).repeat()
    ###prep training###
    Train_Data=datasets['train'].batch(32).repeat()
    #ds_short=Train_Data.take(10)

    #also hard coded for now
    m_file='tmp/mymodel.h5'

    trop_model=keras.models.load_model(m_file,
            custom_objects=tropical_objects)#compile=False
    net_model=trop2dnn(trop_model.input_shape,trop_model.layers)

    f_aff,g_aff=local_linear(trop_model,X)

    #next steps --- define loss, loop through samples, get biggest loss per
    #def loss(F


##loss F,G,x,y
    #return loss for each i



    ##cfx##
    n_greedy_iter=1000
    greedy_data=Train_Data.take(n_greedy_iter)
    net=net_model
    fg_model=trop_model

    ##cfx##

    ##Notes##
        #Throughout, I try to use caps F,G,Poly, 
        # for var names instead of f,g,poly when referring to current 
        # accumulated model pred, and lowercase for
        # preds based on local lin approx to current batch
        #"sig sample iter"--when was this guy added
        #    keys have to be lower for both

        #rule for pmy_polys "plus/minus y" 
        #"my"=-y_pm1  "py"=+y_pm1
        #for my_poly:
        #    y>0 eval g_sig(x)(x), for y<=0 eval f_sig(x)(x)
        # reverse for py_poly

    #db
    bh = peak( greedy_data )


    ###   Init SS Loop   ###
    ss_iter = -1
    Fw = tf.zeros( (1,)+net.input.shape[1:],dtype=net.input.dtype)
    Fb = tf.zeros( (1,1), dtype=net.output.dtype )
    Gw = tf.zeros( (1,)+net.input.shape[1:],dtype=net.input.dtype)
    Gb = tf.zeros( (1,1), dtype=net.output.dtype )
    ad_Greed=ArrayDict({
        'df.w':Fw,#keys have to be lower
        'df.b':Fb,#  to match batch keys
        'dg.w':Gw,
        'dg.b':Gb,
        'delete me':True,
    })
    ###   Init SS Loop   ###

    ad_Packrat=ArrayDict()

    print('Starting greedy loop with ',n_greedy_iter,' iterations')
    t0=time.time()
    for ss_iter, bh in enumerate(greedy_data):

        #loop for xy in bh in data
        x,y=bh['x'],bh['y']
        bh['ss_iter']=ss_iter * np.ones_like(y)
        bh['net(x)'] = net( x )
        fg_x,DfDg_x=eval_f_and_Df( fg_model,  bh['x'] )
        #bh['f.w*x+f.b'],bh['g.w*x+g.b']=fg_x
        bh['f_x'],bh['g_x']=fg_x #okay it happens that df.w*x+df.b = f_x
        bh['fg_polys_x']=tf.stack(fg_x,axis=1)#f,g at sig(x) at x
        #bh['dfdg_Polys_x']=tf.stack(DfDg_x,axis=1) #slightly tricky
        #bh['trop(x)']=bh['df.w*x+df.b'] - bh['dg.w*x+dg.b']
        bh['trop(x)']=bh['f_x'] - bh['g_x']
        f_ll,g_ll=DfDg_x
        bh['df.w'],bh['df.b']=f_ll
        bh['dg.w'],bh['dg.b']=g_ll

        loss_x=calc_greedy_loss(bh,ad_Greed)
        #bh['greedy loss']=loss_x  #handled inside calc_greedy_loss

        ad_Batch=ArrayDict(bh)#just easier to slice
        ind=np.argmax(loss_x)
        new_info=ad_Batch[ ind:ind+1: ]#slice(n,n+1)# keeps batchdim of bsz=1

        if 'delete me' in ad_Greed.keys():#throw away 0 as starting pt
            assert ss_iter==0
            ad_Greed=new_info
            bh0=ad_Batch
        else:#ss_iter>0
            assert ss_iter>0
            ad_Greed.concat(new_info)

        ad_Packrat.concat(ad_Batch)#just keep everything

    print('Finished ',1+ss_iter,' steps of Greedy Sig in ',time.time()-t0)

    y_hot=tf.one_hot(y,depth=2)#db



