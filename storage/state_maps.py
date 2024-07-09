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
from utils import prepare_dirs_and_logger,save_config,make_folders
from config import get_config



'''
"State" applies to the current batch
"Sigma" may be fixed ahead of time

"act" : "activation" : after the relu nonlinearity. input is act_0 output is act_{d+1}
"lin" : "linear output : before the relu nonlinearity. output is lin_{d+1}
Hence for a d=depth many layer network, there are d+1 "activations" and d "linear outputs"
'''


#def pos_cmpt(weight):
#    tf.nn.relu(weight)

pos_cmpt =lambda weight : tf.nn.relu(weight)
neg_cmpt =lambda weight : pos_cmpt(-weight)
abs_cmpt =lambda weight : pos_cmpt(weight)+neg_cmpt(weight)
pos_cmpts=lambda weights:[pos_cmpt(w) for w in weights]
neg_cmpts=lambda weights:[neg_cmpt(w) for w in weights]
abs_cmpts=lambda weights:[abs_cmpt(w) for w in weights]


def apply_keras_layer(X,layer,with_weights=None):
    '''
    Small wrapper to allow proper handling of Misc layers
    and to allow for broadcasting, since we have
    batch_size and sig_size floating around

    reshaping has to be handled within each layer since we allow an arbitrary
    number of leading indicies. The layer type determines how many feature
    dimensions to reshape into
    '''
    if isinstance(layer,keras.layers.Flatten):
        ##Here I'm assuming that this comes from conv2d##
        #flat_shape=X.shape[:-3]+tf.TensorShape(X.shape[-3:].num_elements() )
        flat_shape=[-1]+X.shape[1:-3].as_list()+[X.shape[-3:].num_elements() ]
        return tf.reshape(X,flat_shape)

    else: #fc and conv2d
        return apply_linear_layer(X,layer,with_weights)


def apply_linear_layer(X,layer,with_weights=None):
    '''
    Inputs:
        X,layer,with_weights
    Applies the computation defined by the keras layer,
    but allows the weights to be substituted optionally

    Also, application to X is broadcasted, with the number of feature
    dimensions being determined by layer attributes
    '''


    weights = with_weights or layer.weights

    #if weights is with_weights: #DB
    #    print('LinearLayer passed aux weights')

    #These are the only ones I've verified work#
    if isinstance( layer , keras.layers.Conv2D ): #allow ndim>4 for conv2d

        #not sure which rs approach is best
        rs_X= tf.reshape(X,[-1,]+X.shape[-3:].as_list())
        rs_lin_act = tf.nn.conv2d(rs_X,weights[0], layer.strides,
                        layer.padding.upper()) + weights[1]
        #out_shape=X.shape[:-3]+rs_lin_act.shape[-3:]
        #out_shape=[-1]+X.shape[1:-3]+rs_lin_act.shape[-3:]
        out_shape=[-1]+X.shape[1:-3].as_list()+rs_lin_act.shape[-3:].as_list()
        lin_act=tf.reshape(rs_lin_act, out_shape)

    elif isinstance( layer, keras.layers.Dense ):
        lin_act = weights[1]+ tf.matmul( X, weights[0] )

    else:
        raise ValueError('unexpected layer type',type(layer),
                         'for layer ',layer.name)
    return lin_act



def apply_sig(X,sig):
    '''
    Output format is [batch_size, sig_size,]+data_shape
    where data_shape is shared by the n-1 indicies of
    X.shape and sig.shape

    There are two modes, coordinate-wise and product,
    depending on whether each sig is being applied to a
    single x or to all of them

    This function just lets you handle that. :)
    It'll broadcast if it can, and change dims otherwise

    If the initial broadcast works, that means either
    1) X.shape==sig.shape   (coordinate-wise mul)
    2) X.shape[1]==1   sig.shape[0]==1   X.shape[2:]==sig.shape[2:]
    at least, that's the intended use case
    '''

    try:
        act=X*sig
    except tf.errors.InvalidArgumentError as e:
        #print('no big deal.',e)
        ed_X=tf.expand_dims(X,1)
        ed_sig=tf.expand_dims(sig,0)
        act = ed_X*ed_sig
    return act



def net_at_state_map(X,Sigma,layers,return_preactivations=False):
    act=X
    sig_act=[act] #I'm undecided on   #InputAreActs
    #sig_act=[] #whether include input in activations
    sig_lin=[]
    #lyr_act=[act]
    for sig,lyr in zip(Sigma,layers):
        #print('start layer ',len(sig_act))
        lin=apply_keras_layer(act,lyr)
        act=apply_sig(lin,sig) if not sig is Sigma[-1] else lin
        sig_lin.append(lin)
        sig_act.append(act)

    if return_preactivations:
        return sig_act,sig_lin
    else:
        return sig_act



#Try more numerical stable version
#Still more improvements possible
def fg_recurse(f,g,sig,lyr):
    ## Nonnegligible numerical improvement ##
    #   #(f-g)+g \not\approx f  for large networks
    #   I was getting 0.33 max diff over 43,000 neurons
    #   It seems both f,g can be LARGE compared to f-g

    if not lyr.weights: #is Flatten
        #print('skipping due to flatten')
        f_sig=apply_sig(f,sig)#stylistic decision
        g_sig=apply_sig(g,sig)
        f_next=apply_keras_layer(f_sig,lyr)
        g_next=apply_keras_layer(g_sig,lyr)
        #f_next=apply_keras_layer(f,lyr)
        #g_next=apply_keras_layer(g,lyr)
        return f_next,g_next

    #p=f_sig-g_sig
    p=apply_sig(f-g,sig)#is act for sig=state(x)

    pos_wts=pos_cmpts(lyr.weights)
    neg_wts=neg_cmpts(lyr.weights)
    abs_wts=abs_cmpts(lyr.weights)

    #Apply layers to f,g... try not to take diff
    #fnext=[W+,B+](uf) + [W-,B-](g) + [W+,B+]( (1-u)g )=A1+A2+A3
    #gnext=[W-,B-](uf) + [W+,B+](g) + [W-,B-]( (1-u)g )=B1+B2+B3

    u_f =apply_sig(f,sig)
    uc_g=apply_sig(g,1.-sig)

    #if u_f.ndim>g.ndim:#sig expanded dim
    if len(u_f.shape)>len(g.shape):#sig expanded dim
        _g=tf.expand_dims(g,1)
    else:
        _g=g

    A1=apply_keras_layer( u_f, lyr, with_weights=pos_wts)
    A2=apply_keras_layer(  _g, lyr, with_weights=neg_wts)
    A3=apply_keras_layer(uc_g, lyr, with_weights=pos_wts)

    B1=apply_keras_layer( u_f, lyr, with_weights=neg_wts)
    B2=apply_keras_layer(  _g, lyr, with_weights=pos_wts)
    B3=apply_keras_layer(uc_g, lyr, with_weights=neg_wts)

    #Could drop abs(bias) as well

    f_next=A1 + A2 + A3
    g_next=B1 + B2 + B3

    return f_next,g_next


def trop_polys(X,Sigma,layers):
    #if not X.shape[0]==Sigma[0].shape[0]:
    #    X=tf.expand_dims(X,1)
    #    Sigma=[tf.expand_dims(sig,0) for sig in Sigma]

    f=apply_keras_layer(X,layers[0])
    g=tf.zeros_like(f)
    f_trop,g_trop=[f],[g]

    #Sigma[k] + fg_trop[k] + layers[k+1] --> fg_trop[k+1]
    for sig,lyr in zip(Sigma[:-1],layers[1:]):
        #last Sigma is output label
        #first layer is already applied to init f_0,g_0
        #print('starts with lin ',len(f_trop),':','uses sigshape:',sig.shape)
        #print('\t','and layer ',lyr.name)
        f,g=fg_recurse(f,g,sig,lyr)
        f_trop.append(f) ; g_trop.append(g)
    return f_trop,g_trop



if __name__=='__main__':
    pass



