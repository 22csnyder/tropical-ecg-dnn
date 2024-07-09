from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import get_config
import pickle
#from utils import prepare_dirs_and_logger,save_config

import math
import numpy as np
import os
import json
import sys
import glob2
from itertools import product
from tqdm import trange
import time
import copy

from ArrayDict import ArrayDict
from config import get_config
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

import sympy
from sympy import symbols
from sympy.logic.boolalg import And,Or
from sympy.logic import boolalg
from sympy import Max,Min
from sympy.utilities.lambdify import lambdify
from sympy import preorder_traversal,postorder_traversal

#from standard_data import *
import standard_data
import standard_model
from standard_data import peak,inspect_dataset,tuple_splits
from standard_model import tropical_objects #help load custom layers
from tropical_layers import TropicalSequential

from utils import SH,DIFF,make_folders,tf2np,np_relu
from tboard import file2number
from ArrayDict import ArrayDict


'''

This is supposed to group code snippets that take dnn models
in some form to minmax or andor trees in some form

Definitive best way to do this is unclear

'''
def rescale_weights(weights,activations):
    '''
    Takes wts (weights) and neuron activations
    (thus the last dim must agree
    and zeros the last dims determined by multiplying
    the two arguments after reshaping

    Converts network activations and weights into a list of affine maps
    that compose to give you the network function
    corresponding to the activation pattern


    sigl only corresponds to network states. don't pass in array([1]) for final
    network value
    '''
    wts=weights
    sigl=activations

    assert(len(wts)==len(sigl)+1)

    w0,b0=wts[0]
    wpad=len(w0.shape)-2#last two dim for matrix
    sh_widx=w0.shape[:wpad]
    sig_pad=len(sigl[0].shape)-1

    #Reshaping
    #sig_shapes=[(1,)*wpad+s.shape for s in sigl]
    #rs_X=X.reshape(Xshape)

    def wrs(W):#also works on biases
        new_shape=sh_widx + (1,)*sig_pad + W.shape[wpad:]
        return W.reshape(new_shape)
    rs_weights=[[wrs(W),wrs(b)] for W,b in wts]

    #pretty sure not needed
    #def srs(sig):
    #    new_shape=(1,)*wpad+sig.shape
    #    return sig.reshape(new_shape)
    ##rs_sigl=[srs(sig) for sig in sigl]

    new_weights=[]
    for l in range(len(sigl)):
        w,b=rs_weights[l]#skips last
        sig=sigl[l]
        s1=sig.reshape((1,)*wpad+sig.shape[:-1]+(1,)+sig.shape[-1:])
        s2=sig.reshape((1,)*(wpad+0)+sig.shape)
        new_weights.append([w*s1,b*s2])
    new_weights.append(wts[-1])#Last one unchanged

    return new_weights

def compose_affine(list_affine):
    #func section
    '''
    Takes list of network weights (with abitrary leading indicies)
    and composes them as if they were a composition of affine functions
    without nonlinearities between
    '''
    wts=list_affine
    w0=wts[0][0]
    #xdim=w0.shape[-2]

    #A,b
    #init
    #A=np.eye(xdim)
    #b=np.zeros

    #x*Linear+Bias
    Linear,Bias=wts[0]

    #W,b=wts[1]#test
    for W,b in wts[1:]:
        rsLinear=np.expand_dims(Linear,axis=-1)
        rsBias=np.expand_dims(Bias,axis=-1)
        rsW=np.expand_dims(W,axis=-3)
        Linear=np.sum(rsLinear*rsW, axis=-2)
        Bias=np.sum(rsBias*W,axis=-2)+b
        #Bias=np.sum(rsBias*rsW,axis=-2)+b

    #print 'Affine map is \n'+\
    #    'Linear',Linear,'\n'+\
    #    'Bias',Bias
    return (Linear,Bias)

def AndOr2MinMax(Op):
    if Op is Or:
        return Max#sympy.Max
    elif Op is And:
        return Min#sympy.Min
    else:
        raise ValueError('Either simpy.And or simpy.Or must be passed')
def MinMax2AndOr(Op):
    if Op is Max:
        return Or#sympy.Max
    elif Op is Min:
        return And#sympy.Min
    else:
        raise ValueError('Either simpy.Min or simpy.Max must be passed')

def swap_bool(Op):
    if Op is Or:
        return And
    elif Op is And:
        return Or
    else:
        raise ValueError('Either simpy.And or simpy.Or must be passed')
def swap_minmax(Op):
    if Op is Max:
        return Min
    elif Op is Min:
        return Max
    else:
        raise ValueError('Either simpy.Min or simpy.Max must be passed')


#fc booltree construction
#args:
    # weights,sigl

#def TreeSprout1( weights, sigl , init_op = 'And'):
def TreeSprout1( weights, sigl , init_op ):
    '''
    weights are numpy weights of fully connected network [(W,b),*]
    sigl is list of unique states at each layer. np.hstack(sigl) has unique rows
    '''
    t_start = time.time()


    #assert init_op in [And,Or]
    assert init_op in [sympy.Max,sympy.Min]
    #if not init_op in [And,Or]:
    #    if isinstance(init_op,str):
    #        init_op=getattr(boolalg,init_op)
    #init_op=getattr(boolalg,init_op) if init_op in ['And','Or']
    ##init_op=getattr(sympy,init_op) if init_op in ['Min','Max']

    #0,1,1 is the config I settled on. The rest of it was messing around.
    #WhichSig='All'
    #WhichSig=0

    OrAndPivot=1
        #1 only start last layer with init_op, alternate
        #2 always start each layer with init_op
    LeafSigMethod=1
        #1 sets new_idx_remain_sig=idx_remain_sig[c2_inv==i2]
        #2 LeafSig includes all list_unq_sig[hidden_layer]

    #Either is a fine choice
    #init_op=Or
    #init_op=And

    #sig_str='WS'+str(WhichSig)
    #config_str='WS'+str(WhichSig)+'_OAP'+str(OrAndPivot)+'_LSM'+str(LeafSigMethod)
    #config_str+='_io'+str(init_op)


    #if WhichSig=='All':
    list_unq_sig=sigl
    #elif WhichSig==0:
    #    list_unq_sig=sigl0
    #else:
    #    raise ValueError('WhichSig is ',WhichSig)

    #init
    n_sig=len(list_unq_sig[0])
    depth=len(list_unq_sig)
    Bool_Tree=symbols('0')
    MinMax_Tree=symbols('0')
    leaf=Bool_Tree.atoms().pop()#atoms() returns a set
    leaf_stack={leaf:{
                      'last_operator'  :init_op,#default start
                      #'idx_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
                      'id_mu_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
                      'id_tu_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
                      'hidden_layer'   :depth,#0,...,depth-1(first is dummy)
                      'affine'         :weights[-1],
                     }
               }
    debug_stack=leaf_stack.copy()#shallow
    terminal_leafs={}#so ctrl-f finds it


    print('Entering LeafStack While Loop')
    #LOOP=-1
    while leaf_stack:#until empty
    #    LOOP+=1#DEBUG
        #start loop
        a_leaf=leaf_stack.keys()[0]
        leaf_info=leaf_stack.pop(a_leaf)#while nonempty
        leaf_parent_operator=leaf_info['last_operator']#sympy Max/Min
        #idx_remain_sig=leaf_info['idx_remain_sig']
        id_mu_remain_sig=leaf_info['id_mu_remain_sig']
        id_tu_remain_sig=leaf_info['id_tu_remain_sig']
        ##Increment## hidden_layer -= 1
        hidden_layer=leaf_info['hidden_layer']-1   #hidden layers iterate 0,..,depth-1
        gamma,beta=leaf_info['affine']#current linear map
        #bool_remain_sig=leaf_info['bool_remain_sig']
        #print('leaf:',a_leaf)
        #print('G',gamma,'b',beta,'\n')
        #print('DEBUG','IDX_REMAIN_SIG',idx_remain_sig)
        #leaf_sig=list_unq_sig[hidden_layer][idx_remain_sig]#becomes mu_,tu_leaf_sig
        mu_leaf_sig=list_unq_sig[hidden_layer][id_mu_remain_sig]
        tu_leaf_sig=list_unq_sig[hidden_layer][id_tu_remain_sig]
            #leaf_sig=list_unq_sig[hidden_layer]#simply use all
        #leaf_weights=weights[hidden_layer:]#includes weights feeding into layer
        leaf_weights=[weights[hidden_layer],[gamma,beta]]#includes weights feeding into layer

        gamma_pos=(gamma>0).astype('int').ravel()#strict ineq justified
        gamma_neg=(gamma< 0).astype('int').ravel()
        pos_sig,pos_inv=np.unique(gamma_pos*mu_leaf_sig,axis=0,return_inverse=True)
        neg_sig,neg_inv=np.unique(gamma_neg*tu_leaf_sig,axis=0,return_inverse=True)
        mu_Sig,tu_Sig=pos_sig,neg_sig#db
        #pos_sig,pos_inv=np.unique(gamma_pos*leaf_sig,axis=0,return_inverse=True)
        #neg_sig,neg_inv=np.unique(gamma_neg*leaf_sig,axis=0,return_inverse=True)
        #unq_sig,unq_inv=np.unique(leaf_sig,axis=0,return_inverse=True)

        #split into pos,neg pieces
        #posrelu_gamma+negrelu_gamma=gamma
        posrelu_gamma,posrelu_beta= np_relu( gamma).ravel(), np_relu( beta)
        negrelu_gamma,negrelu_beta=-np_relu(-gamma).ravel(),-np_relu(-beta)
        #mu_gamma#mu_beta
        mu_beta,tu_beta=posrelu_beta,negrelu_beta
        mu_gamma,mu_inv=np.unique(posrelu_gamma*mu_leaf_sig,axis=0,return_inverse=True)
        tu_gamma,tu_inv=np.unique(negrelu_gamma*tu_leaf_sig,axis=0,return_inverse=True)
        ##Careful because entries in mu_leaf_weights not all same shape len
        ##but compose_affine seems to broadcast correctly anyway
        mu_leaf_weights=[weights[hidden_layer],[mu_gamma[...,None],mu_beta]]
        tu_leaf_weights=[weights[hidden_layer],[tu_gamma[...,None],tu_beta]]
        new_mu_Gamma,new_mu_Beta=compose_affine(mu_leaf_weights)#n_unqxnlx1,n_unqx1
        new_tu_Gamma,new_tu_Beta=compose_affine(tu_leaf_weights)


        ##How to decide op order##
        if OrAndPivot==1:#default
            c1_opt= leaf_parent_operator
            c1_op = MinMax2AndOr(c1_opt)
        #elif OrAndPivot==2:#experimental only
        #    c1_op=init_op
        c2_op=swap_bool(c1_op)#always
        c2_opt=swap_minmax(c1_opt)
        #c1_opt=AndOr2MinMax(c1_op)
        #c2_opt=AndOr2MinMax(c2_op)

        if c1_opt is Max:
            #c1_sig,c1_inv=pos_sig,pos_inv#pre v3
            c1_Gamma,c1_Beta,c1_inv=new_mu_Gamma,new_mu_Beta,mu_inv
            c2_Gamma,c2_Beta,c2_inv=new_tu_Gamma,new_tu_Beta,tu_inv
            c1_Sig= pos_sig#keep track of which mu/tau was used
            c2_Sig=-neg_sig
            c1_is_mu_=True
        #elif c1_op is And:
        elif c1_opt is Min:
            c1_Gamma,c1_Beta,c1_inv=new_tu_Gamma,new_tu_Beta,tu_inv
            c2_Gamma,c2_Beta,c2_inv=new_mu_Gamma,new_mu_Beta,mu_inv
            c1_Sig=-neg_sig
            c2_Sig= pos_sig
            c1_is_mu_=False
        else:
            raise ValueError('c1_op is ',c1_op)

        sig2name=lambda a: ''.join([str(b) for b in a])
        c1_sym_list=[]
        c1_opt_list=[]
        for i1,c1_gamma in enumerate(c1_Gamma):
            c1_beta=c1_Beta[i1]
            c1_sig=c1_Sig[i1]
            #c1_sym=sig2name(c1_sig)

            ##Decoupled last operator
            if len(c2_Gamma)>1:
                last_operator=c2_opt
            else:
                last_operator=c1_opt
            ##Only relevant when pairing down c2 by c1##
            #if len(i2_corresp_i1)>1:
            #    last_operator=c2_op
            #else:
            #    last_operator=c1_op

            c2_sym_list=[]
            for i2,c2_gamma in enumerate(c2_Gamma):
                c2_beta=c2_Beta[i2]
                c2_sig=c2_Sig[i2]
                c2_sym=sig2name(c2_sig)
                assert(np.dot(np.abs(c1_sig),np.abs(c2_sig))==0.)

                new_ext=sig2name(c1_sig+c2_sig)
                new_name=new_ext+'.'+str(a_leaf)
                new_node=symbols(new_name)
                c2_sym_list.append(new_node)

                ##New Affine##
                new_gamma=c1_gamma+c2_gamma
                new_beta=c1_beta+c2_beta

                ####new idx####
                if LeafSigMethod==1:#v1
                    #new_idx_remain_sig=idx_remain_sig[c2_inv==i2]
                    if c1_is_mu_:#mu is first coord
                        new_id_mu_remain_sig=id_mu_remain_sig[c1_inv==i1]
                        new_id_tu_remain_sig=id_tu_remain_sig[c2_inv==i2]
                        mu_sig,tu_sig=c1_sig,c2_sig#for debug
                    else:
                        new_id_mu_remain_sig=id_mu_remain_sig[c2_inv==i2]
                        new_id_tu_remain_sig=id_tu_remain_sig[c1_inv==i1]
                        mu_sig,tu_sig=c2_sig,c1_sig#for debug
                elif LeafSigMethod==2:#all idx always
                    #new_idx_remain_sig=idx_remain_sig
                    new_id_mu_remain_sig=id_mu_remain_sig
                    new_id_tu_remain_sig=id_tu_remain_sig
                new_leaf_info={
                                'last_operator'  :last_operator,
                                #'idx_remain_sig' :new_idx_remain_sig,
                                'id_mu_remain_sig' :new_id_mu_remain_sig,
                                'id_tu_remain_sig' :new_id_tu_remain_sig,
                                'affine' :[new_gamma,new_beta],
                                'Sig_sym'           :new_node,
                              }

                if hidden_layer>0:
                    new_leaf_info['hidden_layer']=hidden_layer
                    leaf_stack[new_node]=new_leaf_info
                else:
                    #assert(len(new_idx_remain_sig)==1)#corresp to 1 guy
                    terminal_leafs[new_node]=new_leaf_info

                #debug at top of network
                if hidden_layer>=1:
                    debug_stack[new_node]=new_leaf_info

            c1_sym_list.append( c2_op(*c2_sym_list ) )
            c1_opt_list.append( c2_opt(*c2_sym_list) )
        new_branch    =c1_op(*c1_sym_list)
        new_branch_opt=c1_opt(*c1_opt_list)
        Bool_Tree  =Bool_Tree.subs(a_leaf,new_branch)
        MinMax_Tree=MinMax_Tree.subs(a_leaf,new_branch_opt)


    t_total = time.time() - t_start

    Leafs,List_Leaf=zip(*terminal_leafs.items())#Defines ordering from here on
    ##Make readable##
    alphabet=symbols('a0:%d'%len(Leafs))
    subdict={atom:letter for atom,letter in zip(Leafs,alphabet)}
    abstract_Bool_Tree=Bool_Tree.subs(subdict)
    abstract_MinMax_Tree=MinMax_Tree.subs(subdict)
    abt=abstract_Bool_Tree

    abstract_terminal_leafs = {subdict[k]:v for k,v in terminal_leafs.items()}

    for sig_sym,PoOT_sym in subdict.items():
        leaf=terminal_leafs[sig_sym]
        leaf['PoOT_sym']=PoOT_sym

    print('..Done with calc MinMax_Tree.. took',t_total,'(s)')

    return abstract_MinMax_Tree,abstract_terminal_leafs





if __name__=='__main__':




    pass



