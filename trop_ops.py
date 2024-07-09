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


from tropical_layers import TropicalRational


'''
Various including Tropical Linear Algebra
'''

def tropdot(A,B,trop_sum):
    #intent is last dim A & first dim B
    assert A.shape[-1] == B.shape[0]
    bc_A = A.reshape(A.shape+(1,)*(B.ndim-1))
    return trop_sum(bc_A + B, axis=A.ndim-1)
def mindot(A,B):
    return tropdot(A,B,np.min)
def maxdot(A,B):
    return tropdot(A,B,np.max)

def nat(A,B):
    #calculates A#B
    #denoted with natural symbol like # in notes
    #nat(A,B)_i,k = min_j -Aji+Bjk 
    #note  nat(A,A)<=0 with equality on diagonal
    assert A.ndim==2
    return mindot(-A.T,B)

#def maxproj(A,B):#of cols(B) to colspace(A)
def maxproj(onto_colspace_of_A,of_B):#of cols(B) to colspace(A)
    A=onto_colspace_of_A
    B=of_B
    #Returns B iff B=maxdot(B,C) for some C
    #   C=nat(A,B) will suffice
    #else  B >= maxproj(A,B)  cwise
    return maxdot( A, nat(A,B) )

def row_normalize(mat):
    return mat- np.max(mat, axis=-1,keepdims=True)

def diagmarg(mat):
    #how much can we subtract from diag while remaining row dominant
    #assert mat.shape[0] <= mat.shape[1]#diag behavior
    r,c=mat.shape
    mat=mat[:min(r,c),:c]
    mask=np.ones_like(mat)
    np.fill_diagonal(mask,0.)
    return np.diag(mat) - np.max( mat*mask, -1 )

def row_reduce(F,G,return_H=False):
    #Its okay but unbalanced right now
    #better:  F-Relu(dg-df) , G-Relu(df-dg)
    Fx=np.max(F,axis=-1)[:,None]
    Gx=np.max(G,axis=-1)[:,None]
    N =Fx-Gx
    H=np.maximum(F,N+G)
    Fpme,Gpme = F-H, G-H
    if return_H:
        return Fpme,Gpme,H
    else:
        return Fpme,Gpme


if __name__=='__main__':
    plt.close('all')

    R1=np.random.randint(0,9,(2,2)).astype('float')

    A1=np.array([[5., 1]
                ,[0 ,2]])
    A2=np.array([[3., 1]
                ,[3 ,7]])

    B=np.array([[3., 1],[3,7]])

    F=np.array([[5.,1],[3,0]])
    G=np.array([[3,-1],[4,3]])

    def randcol(k,**kwargs):
        if 'seed' in kwargs.keys():
            np.random.seed(  kwargs.pop('seed') )
        return np.random.randint(0,9,(k,1),**kwargs).astype('float')

    v1=randcol(2,seed=10)
    v2=randcol(2,seed=11)

    mp1=mindot(A1,v1)
    mp2=mindot(A2,v2)



    np.random.seed(22)
    C1=np.random.randint(0,9,[4,4])
    C2=np.random.randint(0,9,[4,4])


    O4=np.array(4*[0,])
    f =np.array([7, 2, 4, -1 ])
    r4=np.arange(4)


    C1r4           = maxdot(C1,r4)
    proj_f         = maxdot( C1,nat(C1,f) )
    proj_C1r4      = maxdot( C1,nat(C1,C1r4) )
    redo_proj_C1r4 = maxproj(C1,C1r4)

    #print('C1\n',C1)
    #print('maxdot(C1,0-5)\n',maxdot(C1,O4-5))
    #print('mindot(C1,0-5)\n',mindot(C1,O4-5))

    #print('C1r4\n',C1r4)
    #print('f\n',f)
    #print('proj_f\n',proj_f)
    #print('proj_C1r4\n',proj_C1r4)
    #print('redo_proj_C1r4\n',redo_proj_C1r4)  #seems to be working

    #B>=maxdot(A, nat(A,B) ) will project columns B to colspace A

    #assert trop_identity < nat(C1,C1) < 0






