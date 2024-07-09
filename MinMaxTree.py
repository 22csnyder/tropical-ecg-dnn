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
import itertools

import sympy
from sympy import symbols
from sympy.logic.boolalg import And,Or
from sympy import Max,Min
from sympy.utilities.lambdify import lambdify
from sympy import preorder_traversal,postorder_traversal

#from standard_data import *
import standard_data
import standard_model
from standard_data import peak,inspect_dataset,tuple_splits
from standard_model import tropical_objects #help load custom layers
from tropical_layers import TropicalSequential

from utils import SH,DIFF,make_folders,tf2np,np_relu,LogOrganizer
from vis_utils import show_ecgs,plt_latex_settings

from tboard import file2number
from ArrayDict import ArrayDict


from GenerateTree import TreeSprout1

'''
Network -> MmMm.. [linear]
'''


#Plan
#split model 2 stages Encode,Classif
#Get states
#wts->MmTree
#samples->pre/postorder traversal (collect idx/node)
#{samples}@node  ->  visualize

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


def swap_bool(Op):
    if Op is Or:
        return And
    elif Op is And:
        return Or
    else:
        raise ValueError('Either simpy.And or simpy.Or must be passed')

def swap_minmax(Op):
    if str(Op)=='Min':
        Out=Max
    elif str(Op)=='Max':
        Out=Min
    else:
        raise ValueError('Expected Max or Min but got ', Op )
    return str(Out) if str(Op)==Op else Out


def preorder_organize(ROOT,max_depth):
    #max_depth = 2
    didx_expr=[[ROOT]] #depth indexed expressions
    didx_peot=[['ROOT']] #depth indexed expressions
    iter_depth=1#ROOT is 0
    #d_sorted_expr = didx_expr[-1]
    while iter_depth<=max_depth:
        new_expr=[]
        new_peot=[]
        #for e in didx_expr[-1]:
        for e,p in zip(didx_expr[-1],didx_peot[-1]):
            e_args=list(e.args)
            new_expr.extend(e_args)
            new_peot.extend([p+'_'+str(e.func)+str(sn) for sn in range(len(e_args))])
        didx_expr.append(new_expr)
        didx_peot.append(new_peot)
        iter_depth+=1
    return didx_expr,didx_peot

#didx_expr,didx_peot = preorder_organize(ROOT,max_depth=2)




if __name__=='__main__':
    plt.close('all')

    #code
    #accuracy, |sig0|, %agreement between tree and orig model
    #74ac,     93s0,    94ag

    #log_dir='./logs/Model_0222_213318_trop1d_smallExp_72acc'

    #log_dir='./logs/Model_0228_003046_trop1d_tinyExp_72acc'#code dev done here
    #log_dir='./logs/Model_0228_020151_trop1d_tinyExp_71.6acc'
    #log_dir='./logs/Model_0301_144742_trop1d_tinyExp'#BAD, res not high enough for sig0 agreement
    #log_dir='./logs/Model_0305_031844_trop1d_tiny_NoAF'    #74ac,93s0,94ag
    #log_dir='./logs/Model_0305_110530_trop1d_tiny_NoAF'    #73ac, 6s0,95ag
    #log_dir='./logs/Model_0305_121818_trop1d_tiny_NoAF'    #72ac,10s0,99.9ag
    #log_dir='./logs/Model_0305_141940_trop1d_tiny_NoAF'    #73ac,36s0,91ag
    #BADag  log_dir='./logs/Model_0305_155022_trop1d_tiny_NoAF_73ac'#73ac,83s0,31ag
    #log_dir='./logs/Model_0305_175053_trop1d_tiny_NoAF_74ac'#74ac,11s0,97ag
    #BADs0 log_dir='./logs/Model_0305_193317_trop1d_tiny_NoAF_73ac' #73ac,457s0
    #BADs0 log_dir='./logs/Model_0305_193317_trop1d_tiny_NoAF_73ac'


    #log_dir='./logs/
    #log_dir='./logs/

    print('log_dir:\n',log_dir)

    #oneday do
    #logo = LogOrganizer(log_dir)

    record_dir=os.path.join(log_dir,'records')
    logic_dir=os.path.join(log_dir,'logic')
    tree_model_dir=os.path.join(log_dir,'tree_model')
    make_folders([record_dir,logic_dir,tree_model_dir])

    ##Intermediate savepoints##
    #fname_upper99=os.path.join(record_dir,'upper99.txt')
    fname_upper99=os.path.join(log_dir,'upper99.txt')#constant across V^V^runs
    fname_sym_abs_Bool_Tree=os.path.join(logic_dir,'sympy_abstract_Bool_Tree_{}.pkl')
    fname_sym_Bool_Tree    =os.path.join(logic_dir,'sympy_Bool_Tree_{}.pkl')
    fname_terminal_leafs   =os.path.join(logic_dir,'terminal_leafs_{}.pkl')
    fname_subs_symbols     =os.path.join(logic_dir,'leaf_substitute_symbols_{}.pkl')
    fname_MinMaxTree    = os.path.join(tree_model_dir,'minmax_tree.pkl')
    fname_TerminalLeafs = os.path.join(tree_model_dir,'terminal_leafs.pkl')
    fname_DfTree        = os.path.join(tree_model_dir,'df_tree.pkl')

    ##Intermediate savepoints##
    if not os.path.exists(logic_dir):
        os.makedirs(logic_dir)

    id_str=str(file2number(log_dir))
    ##Infer Experiment Setup##
    load_model_file=os.path.join(log_dir,'checkpoints/Model_ckpt.h5')
    config_json=os.path.join(log_dir,'params.json')
    with open(config_json,'r') as f:
        load_config=json.load(f)
    datasets,info=getattr(standard_data,load_config['data'])()
    ds_train=datasets['train']
    #ds_input=ds_train.map(lambda e:e['x']).batch(1)
    #modify if taking subset
    try:
        train_size=info.train_size
        test_size=info.test_size
    except:
        train_size=info.splits['train'].num_examples
        test_size=info.splits['test'].num_examples
    #---reference batch for datasets---#
    db_bat=peak(ds_train.batch(25))
    X25=db_bat['x']
    X20=X25[:20]
    X=X20

    model=keras.models.load_model(load_model_file,#pass h5 file
        custom_objects=tropical_objects)


    act_shapes=[L.output_shape for L in model.layers] #ex(None,6,16)
    hidden_act_dims=np.array([np.prod(s[1:]) for s in act_shapes])[:-1]
    #encode_ldx=np.where(hidden_act_dims==hidden_act_dims.min())[0][-1]+1

    #where to grid search
    first_skinny_ldx=np.where(hidden_act_dims==hidden_act_dims.min())[0][0]
    ldx_with_weights=np.where([len(L.weights)>0 for L in model.layers])[0]
    #if non-learning layers after, clump with embedding half
    encode_ldx=ldx_with_weights[ldx_with_weights>first_skinny_ldx][0]-1
    first_decode_ldx = encode_ldx + 1
    encode_layers=model.layers[:first_decode_ldx]
    decode_layers=model.layers[first_decode_ldx:]
    encode_shape =encode_layers[-1].output_shape[1:]
    encode_model=TropicalSequential(encode_layers)
    decode_model=TropicalSequential(decode_layers)


    EL0,EL1,EL2,EL3 = encode_layers[:4]#fast debug ref
    ELneg1,ELneg2,ELneg3,ELneg4 = encode_layers[-4:][::-1]
    DL0,DL1,DL2,DL3 = decode_layers[:4]

    print('Inferred that encode layer was output of layer ',encode_ldx)
    print('Inferred that these are the decode layers:')
    for L in decode_layers:
        print(L.output_shape[1:],' : ',L.name)

    M20=model(X20)
    E20=encode_model(X20)
    D20=decode_model(E20)

    if os.path.exists(fname_upper99):#what is upper activation 99percentile
        print('skipping calculation upper encode data bound')
        upper=np.loadtxt(fname_upper99)
    else:
        EX=[]
        t0=time.time()
        print('calculating upper limits for data encode..')
        for bat in ds_train.batch(500):
            EX.append(encode_model(bat['x']))
        print(time.time()-t0,'seconds')
        cEX=np.concatenate(EX,axis=0)
        upper=np.percentile(cEX,99,axis=0)
        np.savetxt(fname_upper99,upper)
    lower=np.zeros_like(upper)


    #res=15    #dim5
    #res=30    #dim5
    res=50  #dim4
    #res=10   #dim6
    print('Using resolution ',res)

    dx=(upper-lower)/float(res)
    #xi=[np.arange(lower[i],upper[i],dx[i])  for i in range(len(dx))]
    xi= [np.linspace(l,u,res) for l,u in zip(lower,upper)]
    xi= [np.unique(xii) for xii in xi]#for when upper[i]=lower[i].
    print('forming GridX..')
    GridE=np.stack(np.meshgrid(*xi),axis=-1).astype(np.float32)#LxLx2
    rsGridE = GridE.reshape([-1,len(dx)])#batch x dim
    Gr20=rsGridE[:20]

    def denselyr2state(layer):
        #cleanly handle sigmoid terminal layer #though prefer logit=model.output
        tare = layer.activation(0.) if hasattr(layer,'activation') else 0.
        #this will produce a state for "flatten" layers, but to no effect
        return tf.cast( tf.greater(layer.output, tare), tf.float32 )
    sig_list=[denselyr2state(L) for L in decode_layers]
    input2state =keras.Model(inputs=model.inputs       ,outputs=sig_list)

    print('forming decode state model..')
    encode_input=keras.layers.Input(encode_shape)
    dc_act=[encode_input]
    dc_sig=[]

    #learned_decode_layers=filter(decode_layers
    for dc_lyr in decode_layers:
        ##Will include output_dim = n_labels terminal layer
        if dc_lyr.weights:#skip relu layers
            dc_act.append(dc_lyr(dc_act[-1]))
            dc_sig.append( tf.cast(tf.greater( dc_act[-1],0.),tf.float32 ) )
    dc_states=keras.Model(inputs=encode_input,outputs=dc_sig)

    def sigl2Sig(sigl):
        paths =np.concatenate(sigl,axis=-1)
        paths_hidden=paths[:,:-1]
        paths_pred  =paths[:,-1:]
        print('Counting unique Sig..')
        Sig,Inv,Cnts=np.unique(paths_hidden,axis=0,  #147324,256
                               return_inverse=True,
                               return_counts=True)
        IdxPlus =Inv[np.where(paths_pred==1)[0]]
        IdxMinus=Inv[np.where(paths_pred==0)[0]]
        Idx0=np.intersect1d(IdxPlus,IdxMinus)
        Sig0, Cnts0=Sig[Idx0], Cnts[Idx0]  #3301
        return Sig,Sig0

    fnSig  = os.path.join(record_dir,'res'+str(res)+'_Sig.npy')
    fnSig0 = os.path.join(record_dir,'res'+str(res)+'_Sig0.npy')
    if os.path.exists(fnSig):
        print('Loading previously calculated Sig and Sig0')
        Sig =np.load(fnSig)
        Sig0=np.load(fnSig0)
    else:
        print('Calculating and saving Sig,Sig0..')
        grid_sigl=dc_states(rsGridE)
        Sig,Sig0=sigl2Sig(grid_sigl)
        np.save(fnSig,Sig)
        np.save(fnSig0,Sig0)

    print('|Sig|=', Sig.shape[0])
    print('|Sig0|=',Sig0.shape[0])


    #############TRANSITION TO calc_mnist_maps code##########################
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    weights=[tf2np(L.weights) for L in decode_layers if L.weights]#skip relu lyrs
    arch=[b.shape[-1] for w,b in weights[:-1]]#net architecture
    #split up into layers again
    sigl=np.split(Sig, indices_or_sections=np.cumsum(arch)[:-1], axis=-1)
    sigl0=np.split(Sig0, indices_or_sections=np.cumsum(arch)[:-1], axis=-1)


    #############TRANSITION TO calc_mnist_maps code##########################
    #stophere

    MinMax_Tree,terminal_leafs = TreeSprout1(weights,sigl0,init_op=Min)

    with open( fname_MinMaxTree, 'wb') as f:
        pickle.dump(MinMax_Tree,f)
    with open( fname_TerminalLeafs, 'wb') as f:
        pickle.dump(terminal_leafs,f)
    print('continuing with logic eval..')

    tf_Tree={}
    #for leaf in leaf_list:
    for leaf in terminal_leafs.values():
        w,b = leaf['affine']
        w_init = keras.initializers.Constant(value=w)
        b_init = keras.initializers.Constant(value=b)
        lin_lyr= keras.layers.Dense(units=len(b),
            kernel_initializer=w_init, bias_initializer= b_init)
        lin_lyr.leaf=leaf
        #tf_Tree[ leaf['Sig_sym'] ] = lin_lyr(encode_input)
        tf_Tree[ leaf['PoOT_sym'] ] = lin_lyr(encode_input)

    PeOT=list(preorder_traversal(MinMax_Tree))
    PoOT=list(postorder_traversal(MinMax_Tree))
    ROOT=PeOT[0]#=PoOT[-1]

    MaxLayer=keras.layers.Maximum
    MinLayer=keras.layers.Minimum
    for expr in PoOT:
        if expr.func==sympy.Symbol:#already in ValBool
            assert expr in tf_Tree.keys()
            continue
        else:
            #assert expr.func in [sympy.And,sympy.Or]
            assert expr.func in [sympy.Min,sympy.Max]

        klyr=MinLayer() if expr.func==sympy.Min else MaxLayer()
        tf_Tree[expr]=klyr( [tf_Tree[a] for a in expr.args] )




    max_peot_depth=2
    didx_expr,didx_name = preorder_organize(ROOT,max_depth=max_peot_depth)
    peot_expr=list(itertools.chain(*didx_expr))
    peot_name=list(itertools.chain(*didx_name))
    n_nodes=len(peot_expr)


    fname_didx_name=os.path.join(tree_model_dir,'didx_name.pkl')
    fname_didx_expr=os.path.join(tree_model_dir,'didx_expr.pkl')

    with open(fname_didx_name,'wb') as fdd_name, \
         open(fname_didx_expr,'wb') as fdd_expr:
        pickle.dump(didx_name, fdd_name)
        pickle.dump(didx_expr, fdd_expr)



    peot_nodes={peot_name[i]:tf_Tree[peot_expr[i]] for i in range(n_nodes)}
    node_model=keras.Model(inputs=encode_input, outputs=peot_nodes)

    te0=time.time()
    print('Calculating Node Evals..')
    NodeEvals = ArrayDict()
    #for bh in ds_train.batch(500).take(50):
    for bh in ds_train.batch(500):
        bh['e(x)']=encode_model(bh['x'])
        bh.update(node_model(bh['e(x)']))
        bh['N(x)']=model(bh['x'])
        NodeEvals.concat(bh)
    print('..done. Took ',time.time()-te0,'(s)')
    NEx=NodeEvals['x']

    skeys=filter(lambda k:np.sum([s>1 for s in NodeEvals.shape[k]])<=1,NodeEvals.keys())
    df_tree=pd.DataFrame.from_dict({k:NodeEvals[k].ravel() for k in skeys})
    sgn_correct = np.sign(NodeEvals['N(x)'])==np.sign(NodeEvals['ROOT'])#99.4%
    print('Percent aggrement on data: ',np.round(100*np.mean(sgn_correct),2),'%')

    #For now just save whole npy training set
    nmd=tree_model_dir
    np.save(os.path.join(nmd,'dfalgn_TrainX.npy'),NodeEvals['x'])
    np.save(os.path.join(nmd,'dfalgn_TrainY.npy'),NodeEvals['y'])

    df_tree.to_pickle(fname_DfTree)
    df_read=pd.read_pickle(fname_DfTree)
    #print('..finished Logic Code')


    ##still aligned with data. Good. (DB)
#    #Check data agreement
#    data_dir='./data/'
#    proc_data_dir=data_dir+'processed_physionet2017/'
#    trainX=np.load( proc_data_dir+ 'trainX.npy')
#    trainY=np.load( proc_data_dir+ 'trainY.npy')
#    trainX=trainX[trainY!='~']
#    trainX=trainX.reshape([-1,150,1]).astype(np.float32)
#    trainX50=trainX[:50]
#->  #model() and tree_model(encode_model(()) both give same result on X and trainX

    print('finished log_dir:\n\t',log_dir)


