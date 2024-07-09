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
from vis_utils import alpha_ecg,resample_m


'''
Python scripting for working out code for using tree to partition data
for visualization purposes

eventually combine into vis_ecg

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


############new imports############
from MinMaxTree import swap_minmax

def path2node( name_node, nodes_by_layer):
    '''
    name_node is some preorder-name for a node in the tree
    nodes_by_layer is a list of lists of nodes in a layer (didx_name elsewhere)

    usage:
        #named_paths = {nm:path2node(nm,nodes_by_layer) for nm in peot_name}
    '''
    path_from_root = []
    nn=name_node
    for lyr_names in nodes_by_layer:
        between_nodes = filter(lambda ln:nn.startswith(ln), lyr_names )
        #       #assert len(between_nodes)<=1
        #can have _= [u'ROOT_Min6_Max1', u'ROOT_Min6_Max11']
        between_nodes.sort(key=len)
        path_from_root.extend( between_nodes[-1:] )
    return path_from_root


def next_descend(name, nodes_by_layer):#requires didx_name
    if name=='':
        return nodes_by_layer[0]
    current_lyr=np.where([name in lnms for lnms in nodes_by_layer])[0][0]
    if current_lyr == len(nodes_by_layer):
        raise ValueError('node ',name,
             ' already at last layer in nodes_by_layer ',current_lyr)
    nxt_lyr_names=nodes_by_layer[current_lyr+1]
    children = filter( lambda s:s.startswith(name),nxt_lyr_names )
    return children

def np_trues_like( ary ):
    return np.ones_like(ary) > 0.

def Op2ArgOp(lyr_op):
    '''
    lyr_op  either sympy.Max or sympy.Min or 'Max' or 'Min'

    returns pandas "argmax/min" through idxmax or idxmin
    assumes you want to find out which of the columns of a df is optimal
    '''
    if str(lyr_op) == 'Min':
        NodeArgOp = lambda df: df.idxmin(axis='columns')
        #print('lyr_op is Min')
    elif str(lyr_op) == 'Max':
        NodeArgOp = lambda df: df.idxmax(axis='columns')
        #print('lyr_op is Max')
    else:
        raise ValueError('Expected Max or Min but got ', lyr_op )
    return NodeArgOp


def df_unique( df ):
    '''get all unique values in any position in pd.DataFrame.be quick about it'''
    per_col = [ pd.unique(df[c]) for c in df ]
    cat = np.concatenate( per_col).ravel()#ravel probably redundant
    return np.unique(cat)





def Partition_Data_by_Nodes(df_tree, didx_name ,thresh = 800):

    peot_name=filter(lambda c:c.startswith('ROOT'),df_tree.columns)
    peot_name.sort(key=len)#shorter names near output listed first
    init_op = infer_init_op(not_root=peot_name[1])#[0] is root
    peot_tree = df_tree[peot_name]#drop other cols like 'x','label',etc.

    #hold on to your seats
    node_active = pd.DataFrame()#when is each node actually doing s/t
    dfs_ops=[]#list[DF(keys=parent nodes[names opt nodes among children])]
    lyr_op=init_op#decides if is MinMaxMin.. or MaxMinMax..
    nodes_prev=['']#sometimes a node is optimal for no sample besides grid->drop 
    node_active['']=np_trues_like( df_tree.iloc[:,0] )#(m,)

    nodes_prev_opt=['']#opt vs active. dono.
    nodes_prev_act=['']
    fam_opt_nodes=[]
    for lyr_nodes in didx_name:#All possible nodes in next lyr
        #print('nodes prev:',nodes_prev)
        lyr_op = swap_minmax( lyr_op )#does 2swaps before needs to equal init_op
        NodeArgOp = Op2ArgOp(lyr_op)

        df_fam=pd.DataFrame()
        for rent in nodes_prev:
            childs=next_descend(rent,didx_name)
            #calc which node is optimal. Only compare branches sharing parent!
            c_optimal = NodeArgOp( df_tree[childs] )
            df_fam[rent]=c_optimal

            for ch in childs:#each child has exactly 1 parent
                ch_opt = c_optimal==ch
                if ch_opt.any():
                    fam_opt_nodes.append(ch)
                #to be active, the child node must be optimal, AND,
                #  the parent node must be active (and so on)
                node_active[ch]=np.logical_and(node_active[rent] , ch_opt)

        nodes_prev = df_unique( df_fam )
        dfs_ops.append(df_fam)#use to db later

        nodes_prev_opt.extend(nodes_prev)

    print('droping cols: ',dfs_ops[0].columns)
    node_active.drop(columns=dfs_ops[0].columns,inplace=True)#dummy ''=parent(ROOT)
    eff_names = [no for no in node_active if node_active[no].any()]
    reps_count = node_active[eff_names].sum(axis=0)
    vis_nodes=[no for no in eff_names if reps_count[no]>thresh]

    node_pos = peot_tree >=0.

    node_bool={}
    for key in vis_nodes:
        node_bool[key]={
                'pos'  :  node_pos[key],
                'neg'  : ~node_pos[key],
                'act'  :  node_active[key],
                }
    return node_bool, node_active, dfs_ops

def node_bool2data(X,Y,node_bool):
    node_data={}
    for key, Bool in node_bool.items():
        pos_bx,neg_bx,act_bx=Bool['pos'],Bool['neg'],Bool['act']

        pos_X, pos_row_i  = resample_m(X[ pos_bx & act_bx],return_idx=True)
        pos_Y = Y[ pos_bx & act_bx][pos_row_i]
        neg_X, neg_row_i  = resample_m(X[ neg_bx & act_bx],return_idx=True)
        neg_Y = Y[ neg_bx & act_bx][neg_row_i]
        node_data[key] = {'x':(pos_X,neg_X),
                          'y':(pos_Y,neg_Y)}

    return node_data

#        neg_X = X[ neg_bx & act_bx]
#        neg_Y = Y[ neg_bx & act_bx]
#
#        pos_X = X[ pos_bool ]
#        pos_Y = Y[ pos_bool ]
#        pos_Xm , pos_row_i = resample_m(pos_X,return_idx=True)
#        pos_Ym = pos_Y[pos_row_i]
#
#        neg_X = X[ neg_bool ]
#        neg_Y = Y[ neg_bool ]
#        neg_Xm , neg_row_i = resample_m(neg_X,return_idx=True)
#        neg_Ym = neg_Y[neg_row_i]
#
#        dataX= (pos_X,neg_X)
#        dataY= (pos_Y,neg_Y)
#        node_data[key] = {'x':(pos_X,neg_X),
#                          'y':(pos_Y,neg_Y)}
#



#return node_data
#
#
#    for key in vis_nodes:
#        pos_bool       = node_active[key] & node_pos[key]
#        neg_bool       = node_active[key] &(~node_pos[key])
#        node_bool[key] = (pos_bool,neg_bool)
#
#
#    node_data={}
#    for key in vis_nodes:
#        pos_bool = node_active[key] & node_pos[key]
#        neg_bool = node_active[key] &(~node_pos[key])
#
#        if not pos_bool.any():
#            print('WARN:node, ',key,'has empty pos_X')
#        if not neg_bool.any():
#            print('WARN:node, ',key,'has empty neg_X')
#
#        pos_X = X[ pos_bool ]
#        pos_Y = Y[ pos_bool ]
#        pos_Xm , pos_row_i = resample_m(pos_X,return_idx=True)
#        pos_Ym = pos_Y[pos_row_i]
#
#        neg_X = X[ neg_bool ]
#        neg_Y = Y[ neg_bool ]
#        neg_Xm , neg_row_i = resample_m(neg_X,return_idx=True)
#        neg_Ym = neg_Y[neg_row_i]
#
#        dataX= (pos_X,neg_X)
#        dataY= (pos_Y,neg_Y)
#        node_data[key] = {'x':(pos_X,neg_X),
#                          'y':(pos_Y,neg_Y)}
#
#    return node_data, node_active, dfs_ops




if __name__=='__main__':
    #plt.close('all')

    #log_dir='./logs/Model_0228_003046_trop1d_tinyExp_72acc'#code dev done here
    log_dir='./logs/Model_0228_020151_trop1d_tinyExp_71.6acc'

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

    df_tree=pd.read_pickle(logo.fname_DfTree)
    peot_name=filter(lambda k:k.startswith('ROOT'), df_tree.keys())#node names
    peot_name.sort(key=len)#shorter names near output listed first
    init_op = infer_init_op(not_root=peot_name[1])#[0] is root

#    #-----------------------
#    ##DEBUG##
#    print('WARNING DEBUG')#TODO undo
#    dbN=50
#    peot_name=peot_name[:3]
#    df_tree=df_tree.head(dbN)
#    trainX=trainX[:dbN]
#    trainY=trainY[:dbN]
#    #-----------------------


    ##CONFIG##
    #category='ThreshEach'
    category='DataPartition'

    X=trainX
    Y=trainY
    #alp=0.01 #alpha
    alP=40. #
    lnw=0.5  #linewidth  (default is 1.5 for plot())
    #config_str=
    category_str='-'.join([category,'linewid'+str(lnw)])
    descrip=''
    ext='.png'
    figsz=(10,6)
    dpi=400 #100 mpl default
    thresh = 800 #min # of samples to show a node
    ##------##
    lnw=0.5;alP=40.; ext='.png'; figsz=(10,6)#from vis_ecg
    kw_plotfig={'alP':alP,
                'linewidth':lnw,
                'figsize':figsz
               }
    kw_savefig={'bbox_inches':'tight',
                'dpi':dpi,
                #'format':ext[1:]#just +.ext
               }
    ##------##

    tP=time.time()
    node_data,node_active,dfs_ops = Partition_Data_by_Nodes(df_tree, didx_name)
        #def Partition_Data_by_Nodes(df_tree, didx_name ,thresh = 800):
    print('..finished DataPartitioning. (took',time.time()-tP,'s)')


    base_folder=tree_model_dir
    peot_tree = df_tree[ peot_name ]

    vis_folder=os.path.join(base_folder,category_str)
    make_folders([vis_folder])

    #----new partition stuff

    #NamesOpts = find_opt_nodes( df_tree, didx_name, init_op=init_op )

    pn0 = peot_name[5]#db
    pn1 = peot_name[7]#  in didx_name[1]
    pn2 = peot_name[24]# in didx_name[2]


    #hold on to your seats
    node_active = pd.DataFrame()#when is each node actually doing s/t
    dfs_ops=[]#list[DF(keys=parent nodes[names opt nodes among children])]
    lyr_op=init_op#decides if is MinMaxMin.. or MaxMinMax..
    nodes_prev=['']#sometimes a node is optimal for no sample besides grid->drop 
    node_active['']=np_trues_like(trainY)#(m,)

    nodes_prev_opt=['']#opt vs active. dono.
    nodes_prev_act=['']
    fam_opt_nodes=[]
    for lyr_nodes in didx_name:#All possible nodes in next lyr
        #print('nodes prev:',nodes_prev)
        lyr_op = swap_minmax( lyr_op )#does 2swaps before needs to equal init_op
        NodeArgOp = Op2ArgOp(lyr_op)

        df_fam=pd.DataFrame()
        for rent in nodes_prev:
            childs=next_descend(rent,didx_name)
            #calc which node is optimal. Only compare branches sharing parent!
            c_optimal = NodeArgOp( df_tree[childs] )
            df_fam[rent]=c_optimal

            for ch in childs:#each child has exactly 1 parent
                ch_opt = c_optimal==ch
                if ch_opt.any():
                    fam_opt_nodes.append(ch)
                #to be active, the child node must be optimal, AND,
                #  the parent node must be active (and so on)
                node_active[ch]=np.logical_and(node_active[rent] , ch_opt)

        nodes_prev = df_unique( df_fam )
        dfs_ops.append(df_fam)#use to db later

        nodes_prev_opt.extend(nodes_prev)

    #print('droping cols: ',dfs_ops[0].columns)
    node_active.drop(columns=dfs_ops[0].columns,inplace=True)#dummy ''=parent(ROOT)

    eff_names = [no for no in node_active if node_active[no].any()]
    reps_count = node_active[eff_names].sum(axis=0)
    vis_nodes=[no for no in eff_names if reps_count[no]>thresh]

    node_pos = peot_tree >=0.

    for key in vis_nodes:
        pos_bx  =  node_pos[key]
        neg_bx  = ~node_pos[key]
        act_bx  =  node_active[key]

        pos_X = X[ pos_bx & act_bx]
        neg_X = X[ neg_bx & act_bx]
        pos_Y = Y[ pos_bx & act_bx]
        neg_Y = Y[ neg_bx & act_bx]

        pos_L=len(pos_X)
        neg_L=len(neg_X)

        if min(len(pos_X),len(neg_X))==0:
            print('WARNING:pretty weird, ',pos_L,' classif pos',
                  ' and ',neg_L,' classif neg by node ',key)

        pos_X = resample_m( pos_X )
        neg_X = resample_m( neg_X )
        pos_info= id_str+'-'+key+'pos_curves'+str(pos_L)+descrip
        neg_info= id_str+'-'+key+'neg_curves'+str(neg_L)+descrip

        if len(pos_X)>0:
            pos_fname=os.path.join(vis_folder,pos_info)
            pos_fig,pos_ax=alpha_ecg( pos_X, **kw_plotfig)
            pos_fig.savefig(pos_fname+ext,**kw_savefig)
            plt.close(pos_fig)

        if len(neg_X)>0:
            neg_fname=os.path.join(vis_folder,neg_info)
            neg_fig,neg_ax=alpha_ecg( neg_X, **kw_plotfig)
            neg_fig.savefig(neg_fname+ext,**kw_savefig)
            plt.close(neg_fig)


    print('finished log_dir:\n\t',log_dir)




###Old Code###

#    #-----------------------
#    ##DEBUG##
#    print('WARNING DEBUG')#TODO undo
#    dbN=50
#    peot_name=peot_name[:3]
#    df_tree=df_tree.head(dbN)
#    trainX=trainX[:dbN]
#    trainY=trainY[:dbN]
#    #-----------------------
