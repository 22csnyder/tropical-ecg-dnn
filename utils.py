from __future__ import print_function
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import shutil
import sys
import math
import json
import logging
import numpy as np
#from PIL import Image
from datetime import datetime
import pandas as pd
import logging

#np_softplus=lambda x:np.log(1+np.exp(x))
#np_lrelu=lambda x:np.minimum(0.2*x,0)+np.maximum(0.,x)
np_relu=lambda x:np.maximum(0.,x)
#np_exp=np.exp
#np_elu=lambda x: x*(0.5)*(np.sign(x)+1) + (np.exp(x)-1.)*(0.5)*(1-np.sign(x))#Correct
#np_tanh=np.tanh
#np_sigmoid=lambda x: 1./(1.+np.exp(-x))

relu=tf.nn.relu
logit_bxe=tf.losses.BinaryCrossentropy(from_logits=True)
hinge=lambda y_wx: relu(1 - y_wx)
fg_diff=lambda fg:fg[0]-fg[1]
def f_minus_g(fg_obj):
    #if isinstance(fg_obj,list):
    #    assert len(fg_obj)==2
    if len(fg_obj)==2:#covers lists,tuples,arrays with leading dim
        return fg_obj[0] - fg_obj[1]

    #look for first index with a 2 as shape
    #dont use batch_size=2 okay please?
    shape=list(fg_obj.shape)
    shape[0]=0#just in case batch_size=2. I got you.
    fg_ind=shape.index(2)#returns first index of value 2
    f,g = tf.split(fg_obj,num_or_size_splits=2,axis=fg_ind)
    return f - g  #bhsz x 1 



##---- Used for inner products between batches X and batches X* (dual) ---##
##T1,T2 same shape:  batchx(stuff)
#dotproduct over stuff to return either
#batchx1       <->  inner
#batchxbatchx1 <->  outer
##Now allows T2.shape[-1]==2, to compute multiple fg filters at once
#Intended Input shapes for linear functionals::
# T1   bsz1 x Input_Shape x 1
# T2   bsz2 x Input_Shape x 2
#   for "inner" need bsz1==bsz2
def batch_inner(T1,T2):
    dots=tf.reduce_sum(T1*T2,axis=range(1,T1.ndim))
    return tf.expand_dims(dots,-1)
def batch_outer(T1,T2):
    edT1=tf.expand_dims(T1,1)
    edT2=tf.expand_dims(T2,0)
    T1T2=edT1*edT2
    #return tf.reduce_sum( T1T2 , axis=range(2,T1T2.ndim-1) )
    allpairs_dots=tf.reduce_sum( T1T2 , axis=range(2,edT1.ndim) )
    return tf.expand_dims(allpairs_dots,-1)


def infer_dot_axes(T1,T2):
    #The last dimensions refer to input shape and should be shared
    #never product sum over first dimension (batch)


    #ndim1,ndim2=T1.ndim,T2.ndim
    ndim1,ndim2=tf.rank(T1),tf.rank(T2)
    mindim=min(ndim1,ndim2)

    #assert T1.shape[-1]==T2.shape[-1] #else return ind=0
    ind=1
    while T1.shape[-ind]==T2.shape[-ind]:
        ind+=1
        if ind==mindim:
            break #never sum over either batch
    #ind-=1
    idxrng=range(-1,-ind,-1)
    idx0=idxrng[-1]
    assert EQUAL(T1.shape[idx0:],T2.shape[idx0:] )
    #assert EQUAL(T1.shape[-ind:],T2.shape[-ind:] )
    return idxrng

def right_pad_shapes(s1,s2):
    s1,s2=list(s1),list(s2)#case of tuple, tf.TensorShape
    l1,l2=len(s1) ,len(s2)
    L=max(l1,l2)
    for i in range(L):
        if   i>=l1:
            s1+=[1]
        elif i>=l2:
            s2+=[1]
        else:
            assert(s1[i]==s2[i])#will broadcast#
    return s1,s2


def outer_inner(T1,T2):
    '''product sum over all of the dims that are trailing and shared
        outerproduct over leading indicies that are not shared e.g. batch'''
    ax=infer_dot_axes(T1,T2)
    return tf.tensordot(T1,T2,axes=[ax,ax])
def cwise_inner(T1,T2):
    '''scalar multiply coordinatewise over batch indicies
        inner product over trailing shared indicies'''

    #This will have to do for now
    #Have to be able to broadcast by expanding dims to the right
    #while T1.ndim<
    ax=infer_dot_axes(T1,T2)
    input_shape=T1.shape[ax[-1]:]#or T2, same thing
    s1=T1.shape[:ax[-1]]
    s2=T2.shape[:ax[-1]]

    if len(s1)!=len(s2):
        s1,s2=right_pad_shapes(s1,s2)
        T1=tf.reshape(T1,s1+input_shape)
        T2=tf.reshape(T2,s2+input_shape)
    T1T2=T1*T2
    return tf.reduce_sum(T1T2,axis=ax)


    #sh1,sh2=T1.shape[-mindim:],T2.shape[-mindim:]
    #j=mindim


#    edT1=tf.expand_dims(T1,1)
#    edT2=tf.expand_dims(T2,0)
#    allpairs_dots=tf.reduce_sum( edT1*edT2, axis=range(2,edT1.ndim) )

def tf2np(wts):
    if hasattr(wts,'numpy'):
        return wts.numpy()
    else:
        return [w.numpy() for w in wts]

###debugging and interactive utilities###
def SH(stuff):#debug tool to quickly see shapes of stuff
    try:
        print(stuff.shape)
    except:
        try:
            #print([T.shape for T in stuff])
            for i,T in enumerate(stuff):
                print(i,' : ',T.shape)
        except:
            for k,v in stuff.items():
                print(k,' : ',v.shape)

def DIFF(T1,T2):
    return tf.reduce_max(tf.abs(T1-T2)).numpy()
def EQUAL(T1,T2):
    return tf.reduce_all(T1==T2).numpy()


#sometimes useful for counting unique binary activations
def tally_labels(attr):
    '''
    inputs
    attr: dataframe of label attributes

    returns
    df2 : dataframe with each row a unique label combination that occurs in the
    dataset. The index is a unique 'ID' that corresp to that label combination
    real_pdf: dataframe with index='ID' and value is the probability of that
    label combination
    '''
    df2=attr.drop_duplicates()
    df2 = df2.reset_index(drop = True).reset_index()
    df2=df2.rename(columns = {'index':'ID'})
    real_data_id=pd.merge(attr,df2)
    real_counts = pd.value_counts(real_data_id['ID'])
    real_pdf=real_counts/len(attr)
    return df2,real_pdf



def make_folders(folder_list):
    for path in folder_list:
        if not os.path.exists(path):
            os.makedirs(path)


###Some more sane way of handling folders

class LogOrganizer(object):
    categories=['records','logic','tree_model']

    def __init__(self,log_dir):
        self.log_dir=log_dir
        self.subfolders=[]

        for cat in self.categories:
            sf=os.path.join(self.log_dir,cat)
            attr = cat +'_dir'
            self.subfolders.append(sf)
            setattr(self,attr,sf)
        make_folders(self.subfolders)

        self.define_msc_files()
        #add additional ad hoc after init


    def define_msc_files(self):#genough for now
        ##Intermediate savepoints##
        #fname_upper99=os.path.join(record_dir,'upper99.txt')
        self.fname_upper99=os.path.join(self.log_dir,'upper99.txt')#constant across V^V^runs

        #fname_gridX  =os.path.join(record_dir,'gridX.npy')
        #self.fname_Grid_Sig =os.path.join(self.record_dir,'res{}_Sig.npy')
        #self.fname_Grid_Sig0=os.path.join(self.record_dir,'res{}_Sig0.npy')
        #fname_GammaTree=os.path.join(record_dir,'GammaTree.npy')
        #fname_BetaTree =os.path.join(record_dir,'BetaTree.npy')

        #fname_sym_abs_Bool_Tree=os.path.join(logic_dir,'sympy_abstract_Bool_Tree_{}.pkl')
        #fname_sym_Bool_Tree    =os.path.join(logic_dir,'sympy_Bool_Tree_{}.pkl')
        #fname_terminal_leafs   =os.path.join(logic_dir,'terminal_leafs_{}.pkl')
        #fname_subs_symbols     =os.path.join(logic_dir,'leaf_substitute_symbols_{}.pkl')

        self.fname_MinMaxTree    = os.path.join(self.tree_model_dir,'minmax_tree.pkl')
        self.fname_TerminalLeafs = os.path.join(self.tree_model_dir,'terminal_leafs.pkl')
        self.fname_DfTree        = os.path.join(self.tree_model_dir,'df_tree.pkl')

        self.fname_didx_name     = os.path.join(self.tree_model_dir,'didx_name.pkl')
        self.fname_didx_expr     = os.path.join(self.tree_model_dir,'didx_expr.pkl')

    #def add_subfolder(sf_string):



def old_get_model_dir(dataset,logs,descrip=''):
    model_name = "{}_{}".format(dataset, get_time())
    model_dir = os.path.join(logs,model_name)
    if descrip:
        model_dir+='_'+descrip
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def get_model_dir(config):
    #Each run gets a new model_name and model_dir
    #config.model_name = "{}_{}".format(config.dataset, get_time())
    config.model_name = "{}_{}".format(config.prefix, get_time())
    if config.descrip:
        config.model_name+='_'+config.descrip
    model_dir = os.path.join(config.log_dir,config.model_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def prepare_dirs_and_logger(config):
    config.model_dir=get_model_dir(config)

    config.log_code_dir=os.path.join(config.model_dir,'code')
    #if not config.load_path:
    if config.load_path is not config.model_dir:
        for path in [config.log_dir, config.model_dir,
                     config.log_code_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        #Copy python code in directory into model_dir/code for future reference:
        code_dir=os.path.dirname(os.path.realpath(sys.argv[0]))
        model_files = [f for f in listdir(code_dir) if isfile(join(code_dir, f))]
        for f in model_files:
            if f.endswith('.py'):
                shutil.copy2(f,config.log_code_dir)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

##Probably broken in tf2.0
#def get_available_gpus():
#    from tensorflow.python.client import device_lib
#    local_device_protos = device_lib.list_local_devices()
#    return [x.name for x in local_device_protos if x.device_type=='GPU']

###Probably broken in tf2.0
#def distribute_input_data(data_loader,num_gpu):
#    '''
#    data_loader is a dictionary of tensors that are fed into our model
#
#    This function takes that dictionary of n*batch_size dimension tensors
#    and breaks it up into n dictionaries with the same key of tensors with
#    dimension batch_size. One is given to each gpu
#    '''
#    if num_gpu==0:
#        return {'/cpu:0':data_loader}
#
#    gpus=get_available_gpus()
#    if num_gpu > len(gpus):
#        raise ValueError('number of gpus specified={}, more than gpus available={}'.format(num_gpu,len(gpus)))
#
#    gpus=gpus[:num_gpu]
#
#
#    data_by_gpu={g:{} for g in gpus}
#    for key,value in data_loader.items():
#        spl_vals=tf.split(value,num_gpu)
#        for gpu,val in zip(gpus,spl_vals):
#            data_by_gpu[gpu][key]=val
#
#    return data_by_gpu



