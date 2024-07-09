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
from utils import prepare_dirs_and_logger,save_config

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

from utils import SH,DIFF,make_folders,tf2np,np_relu, LogOrganizer
from tboard import file2number
from ArrayDict import ArrayDict

import vis_utils
from vis_utils import alpha_ecg,resample_m,plt_latex_settings
from vis_utils import sgnsep_ecgs, ovrlyd_ecgs

from partition_ecg import Partition_Data_by_Nodes,node_bool2data

'''
Python scripting for visualizing collections of ecgs.
Works on an ecg array level,without caring about the underlying network


Intention is for MinMaxTree.py to be run already.

'''


'''
TODO
report size of dataclusters, not size of resampling in filename

'''




def infer_init_op(not_root):
    loc_max = not_root.find('Max')
    loc_min = not_root.find('Min')

    if 0<=loc_max<loc_min  or  loc_min<0<loc_max:
        return 'Max'
    elif 0<=loc_min<loc_max  or  loc_max<0<loc_min:
        return 'Min'
    else:
        raise ValueError('This is beyond me.shouldnt happen I think.','\n',
                         'recieved ', loc_max,'\n',loc_min,'\n',not_root)


def node2func(name):
    '''returns the op whichs acts on the node and siblings to result in parent '''
    name=str(name)
    assert 'ROOT' in name
    #Mm = name[-4:-1]#not always safe,  'ROOT_Min1_Max28'
    Options=['Min','Max']
    op_rlocs = [name.rfind(op) for op in Options]
    rgt_loc=np.max(op_rlocs)#pos
    if rgt_loc == -1:#ROOT
        return None
    else:
        Mm = name[rgt_loc:rgt_loc+3]
    return Mm


    #BADag  log_dir='./logs/Model_0305_155022_trop1d_tiny_NoAF_73ac'#73ac,83s0,31ag
if __name__=='__main__':
    plt.close('all')
    #fv=finished visualization (can be time consuming, dep. on process)
    #log_dir='./logs/Model_0228_003046_trop1d_tinyExp_72acc'#72ac,35s0 #*
    #log_dir='./logs/Model_0228_020151_trop1d_tinyExp_71.6acc' #*

##!$
#    log_dir='./logs/Model_0305_031844_trop1d_tiny_NoAF'#74ac,93s0,94ag #*
    #log_dir='./logs/Model_0305_110530_trop1d_tiny_NoAF' #73ac, 6s0,95ag #*
#    log_dir='./logs/Model_0305_121818_trop1d_tiny_NoAF'#72ac,10s0,99.9ag,fv #*
#    log_dir='./logs/Model_0305_141940_trop1d_tiny_NoAF'  #73ac,36s0,91ag #*
    #log_dir='./logs/Model_0305_175053_trop1d_tiny_NoAF_74ac'#74ac,11s0,97ag,fv #*
    log_dir='./logs/Model_0305_193317_trop1d_tiny_NoAF_73ac_todovis'

##!$

    print('log_dir:\n\t',log_dir)

    id_str=str(file2number(log_dir))
    logo = LogOrganizer(log_dir)


    tree_model_dir=logo.tree_model_dir
    tmd=tree_model_dir
    logo.fname_MinMaxTree
    logo.fname_TerminalLeafs
    logo.fname_DfTree

    with open(logo.fname_MinMaxTree,'rb') as file_tree, \
         open(logo.fname_TerminalLeafs,'rb') as file_leaf:
        minmax_tree=pickle.load(file_tree)
        terminal_leafs=pickle.load(file_leaf)


    with open(logo.fname_didx_name,'rb') as fdd_name, \
         open(logo.fname_didx_expr,'rb') as fdd_expr:
        didx_name=pickle.load(fdd_name)
        didx_expr=pickle.load(fdd_expr)


    trainX=np.load(os.path.join(tmd,'dfalgn_TrainX.npy'))
    trainY=np.load(os.path.join(tmd,'dfalgn_TrainY.npy'))
    trainX=np.squeeze(trainX)
    trainY=np.squeeze(trainY)
##in future may instead use standard_data to load ecgs (should be equiv), e.g.,
##    load_model_file=os.path.join(log_dir,'checkpoints/Model_ckpt.h5')
##    config_json=os.path.join(log_dir,'params.json')
##    with open(config_json,'r') as f:
##        load_config=json.load(f)
##    datasets,info=getattr(standard_data,load_config['data'])()
##    ds_train=datasets['train']

    df_tree=pd.read_pickle(logo.fname_DfTree)
    peot_name=filter(lambda k:k.startswith('ROOT'), df_tree.keys())#node names
    peot_name.sort(key=len)#shorter names near output listed first
    init_op = infer_init_op(not_root=peot_name[1])#[0] is root


    ##CONFIG##
    descrip=''
    descrip='Mar6'
    #category='ThreshEach'
    category='DataPartition'
    cat=category[:3]
    thresh = 800 #min # of samples to show a node
    X=trainX
    Y=trainY
    ext='.png'
    figsz=(10,6)
    dpi=400 #100 mpl default
    kw_plotfig={
                'linewidth':.5,#1.5 is mpl default
                'figsize':figsz,
               }
    kw_savefig={'bbox_inches':'tight',
                'dpi':dpi,
                #'format':ext[1:]#just +.ext
               }
    ##------##

    base_folder=tree_model_dir
    peot_tree = df_tree[ peot_name ]
    vis_folder=os.path.join(base_folder,category)
    make_folders([vis_folder])
    node_pos = peot_tree >=0.

    #how to associate which ecgs to which nodes
    if category == 'DataPartition':
        print('Starting \"Partition_Data_by_Nodes\"..')
        tP=time.time()
        node_bool,node_active,dfs_ops = Partition_Data_by_Nodes(df_tree,didx_name,thresh)
        node_data = node_bool2data(trainX, trainY, node_bool)

            #def Partition_Data_by_Nodes(df_tree, didx_name ,thresh = 800):
        vis_nodes=node_data.keys()
        print('\t..finished DataPartitioning. (took',time.time()-tP,'s)')

    elif category=='ThreshEach':
        node_data={}
        for key in peot_name:#no resample_m I guess for now
            pos_X = X[ node_pos[key]]
            neg_X = X[~node_pos[key]]
            pos_Y = Y[ node_pos[key]]
            neg_Y = Y[~node_pos[key]]

            node_data[key]={'x'  :(pos_X,neg_X),
                            'y'  : (pos_Y,neg_Y),
                            #'len':(pos_L,neg_L),
                           }
        vis_nodes=peot_name
    vis_nodes.sort(key=len)#len order is also depth order


    #Now call vis methods on each (pos_X,neg_X) pair in node_data.values
    print('Beginning Node Plotting Loop, model=',category,'..')
    #plt_latex_settings()#if want
    for key in vis_nodes:
        print('NODE:',key)
        xynd=node_data[key]

        pos_X,neg_X = xynd['x']
        pos_Y,neg_Y = xynd['y']
        func=node2func(key)
        pos_L,neg_L=map(len,xynd['x'])

        plt_types=['pos','neg','ovr']
        tpe_L={'pos':pos_L,'neg':neg_L,'ovr':pos_L+neg_L}
        fmt_info=id_str+'-'+key+'{}_'+str(cat)+'{}'+descrip
        info={k:fmt_info.format(k,tpe_L[k]) for k in plt_types }
            #pos_info= id_str+'-'+key+'pos_'+str(cat)+str(pos_L)+descrip
            #neg_info= id_str+'-'+key+'neg_'+str(cat)+str(neg_L)+descrip
        fnames={k:os.path.join(vis_folder,info[k]) for k in plt_types}
            #ovr_fname=os.path.join(vis_folder,info['ovr'])
            #pos_fname=os.path.join(vis_folder,info['pos'])
            #neg_fname=os.path.join(vis_folder,info['neg'])

        #Plotting
        ovr_fig,ovr_ax  = ovrlyd_ecgs(pos_X,neg_X ,func=func,**kw_plotfig )
        figS,axeS = sgnsep_ecgs(pos_X,neg_X ,func=func,**kw_plotfig )
        pos_fig,neg_fig=figS
        pos_ax ,neg_ax =axeS
        figs={'pos':pos_fig,'neg':neg_fig,'ovr':ovr_fig}

        for tpe in plt_types:
            tpe_fig=figs[tpe]
            if tpe_fig:
                tpe_fig.savefig( fnames[tpe]+ext,**kw_savefig )

                plt.close(tpe_fig)


    print('..finished plotting')
    print('finished log_dir:\n\t',log_dir)#last line file



