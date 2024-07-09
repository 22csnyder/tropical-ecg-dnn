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
from utils import SH,DIFF,EQUAL #db handy

from config import get_config

import standard_data
import standard_model

#from standard_data import *
from standard_data import peak,inspect_dataset,tuple_splits
from standard_model import logit_bxe

from state_maps import pos_cmpt,neg_cmpt,abs_cmpt,pos_cmpts,neg_cmpts,abs_cmpts
from state_maps import (apply_linear_layer,apply_sig,apply_keras_layer,
                        fg_recurse,net_at_state_map,trop_polys
                       )

from ArrayDict import ArrayDict#gather step

'''
A first attempt at reducing Sig size
and everything involved
'''


if __name__=='__main__':

#-------------Config and Log Handling--------------#
    config,_=get_config()#arg parsing
    prepare_dirs_and_logger(config)
    save_config(config)

    #print('model_dir:',config.model_dir)
    #data_fn=get_toy_data(config.dataset)
    #experiment=Experiment(config,data_fn)
    #return experiment

    # ~ ~ Encouraged to use these folder names ~ ~ #
    model_dir=config.model_dir
    bhsz= config.batch_size
    #model_name=os.path.join(checkpoint_dir,'Model')
    #record_dir=os.path.join(config.model_dir,'records')
    #summary_dir=os.path.join(model_dir,'summaries')
    checkpoint_dir=os.path.join(model_dir,'checkpoints')
    model_file=os.path.join(checkpoint_dir,'Model_ckpt.h5')
    make_folders([checkpoint_dir])#,summary_dir])
    print('[*] Model File:',model_file)



#-------------Load Stuff --------------#
    #would like to load the same dataset
    ###TODO (make not Hardcoded)
    datasets,info=standard_data.load_mnist()

    ds_train=datasets['train']
    ds_X=ds_train.map(lambda e:e['x'])
    #ds_input=ds_train.map(lambda e:e['x']).batch(1)

    ##modify if taking subset
    #train_size=info.splits['train'].num_examples
    #test_size=info.splits['test'].num_examples


    #---reference batch for datasets---#
    db_bat=peak(ds_train.batch(25))
    X25=db_bat['x']
    X20=X25[:20]

    load_path='./logs/Model_1023_132404/checkpoints/Model_ckpt.h5'
    Net=keras.models.load_model(load_path)

#----------db trop ----------#
    layers=Net.layers
    L0,L1,L2=layers[:3]
    Llast=layers[-1]

    K_flatten=keras.layers.Flatten()


    Input=Net.input

    def lyr2state(layer):
        #will need to modify to do maxpool
        #cleanly handle sigmoid terminal layer #though prefer logit=model.output
        tare = layer.activation(0.) if hasattr(layer,'activation') else 0.
        #Yes, this will produce a state for "flatten" layers, but to no effect
        return tf.cast( tf.greater(layer.output, tare), tf.float32 )
    sig_list=[lyr2state(L) for L in Net.layers]


    State=keras.Model(inputs=Net.inputs,outputs=sig_list)
    Sigma20=State(X20)
    Sigma25=State(X25)
    Sigma=Sigma25 ##db config##
    X=X20         ##db config##

    Acts=keras.Model(inputs=Net.inputs,
         outputs=Net.inputs+[L.output for L in Net.layers])#InputAreActs

    #interesting#seems to cache on repeated calls
    #@tf.function
    #def fl_State(X):
    #    return [K_flatten(sig) for sig in State(X)]


    ds_short=ds_train.take(1000)
    bh_short=ds_short.batch(bhsz)
    bh_train=ds_train.batch(bhsz)
    bh_test=datasets['test'].batch(bhsz)
    bh_x=ds_X.take(1000).batch(bhsz)


    #ds_lrnSig=ds_train.take(500)
    ds_lrnSig=ds_train.take(5000)

    ###This is filter based on label, but should filter based on Net()
    bh_poslrn=ds_lrnSig.filter(lambda e:tf.equal(e['y'],1)).batch(bhsz)
    bh_neglrn=ds_lrnSig.filter(lambda e:tf.equal(e['y'],0)).batch(bhsz)



    def gather_sig(dataset):
        bh_sigl=[]
        for bh in dataset:
            bh_sigl.append([ s.numpy() for s in State(bh['x']) ])
        lrnSig=map(np.concatenate,zip(*bh_sigl) )
        return lrnSig


    ##Gather sig
    pos_Sig=gather_sig(bh_poslrn)
    neg_Sig=gather_sig(bh_neglrn)

    ed_pos_Sig=[tf.expand_dims(sig,0) for sig in pos_Sig]
    ed_neg_Sig=[tf.expand_dims(sig,0) for sig in neg_Sig]
    ed_Input=tf.expand_dims(Input,1)


    pos_TropPolys = keras.Model(inputs=Net.inputs,
            outputs=[poly[-1] for poly in trop_polys(ed_Input,ed_pos_Sig,layers)]  )
    neg_TropPolys = keras.Model(inputs=Net.inputs,
            outputs=[poly[-1] for poly in trop_polys(ed_Input,ed_neg_Sig,layers)]  )

    #quick hack for demo
    Xps = np.concatenate([ bh['x'] for bh in bh_poslrn])
    Xng = np.concatenate([ bh['x'] for bh in bh_neglrn])

    FpsXng,GpsXng=pos_TropPolys(Xng)
    FngXps,GngXps=neg_TropPolys(Xps)


    am_FpsXng=np.argmax(FpsXng , axis=1) #about 95/250 unique
    am_GngXps=np.argmax(GngXps , axis=1) #about 93/250 unique

    print('n unique achieving max',len(np.unique(am_FpsXng)),
          len(np.unique(am_GngXps)), 'out of ',len(Xps)+len(Xng))

    #len(np.unique(am_FpsXng))


    stophere########################

#    pos_Fsig,pos_Gsig=pos_TropPolys(Net.input)
#    neg_Fsig,neg_Gsig=neg_TropPolys(Net.input)
#
#    pos_F=tf.reduce_max(pos_Fsig,axis=1)
#    pos_G=tf.reduce_max(pos_Gsig,axis=1)
#
#    neg_F=tf.reduce_max(neg_Fsig,axis=1)
#    neg_G=tf.reduce_max(neg_Gsig,axis=1)


    #pos_Trop = keras.Model(inputs=Net.inputs, outputs=pos_F - pos_G )


    def gather(Fun,dataset):
        bh_array=[]
        for bh in dataset:
            bh_array.append( Fun(bh['x']) )
        return np.concatenate(bh_array)





    ##def##
    bh_lrnsig=ds_train.take(300).batch(bhsz)#make val
    ##def##

    #bh_short_test

    Gatherer=ArrayDict()
    gather_fl_states=[]
    gather_x=[]
    bh_sigl=[]
    print('Gather Sig as numpy')
    for batch in bh_short:
        Gatherer.concat(batch)
        bh_sigl.append([ s.numpy() for s in State(batch['x']) ])
        #bh_sigl.append([ fs.numpy() for fs in fl_State(x) ])

    lrn_idx=Gatherer['idx']
    lrn_Sigma=map(np.concatenate,zip(*bh_sigl) )



    ###db cfx###
    #trop_Sigma=Sigma20  # good for 68% actually
    trop_Sigma=lrn_Sigma #81%    #2s/bh


    ed_Sigma=[tf.expand_dims(sig,0) for sig in trop_Sigma]
    ed_Input=tf.expand_dims(Input,1)
    TropPolys = keras.Model(inputs=Net.inputs,
            outputs=[poly[-1] for poly in trop_polys(ed_Input,ed_Sigma,layers)]  )

    Fsig,Gsig=TropPolys(Net.input)
    F=tf.reduce_max(Fsig,axis=1)
    G=tf.reduce_max(Gsig,axis=1)
    Trop = keras.Model(inputs=Net.inputs, outputs=F - G )

    ###Evaluate###
    train_tuple,test_tuple=tuple_splits(datasets)
    #bh_train_tuple=train_tuple.batch(bhsz).repeat()
    bh_test_tuple = test_tuple.batch(bhsz)


    Trop.compile(
                #optimizer=opt,
                loss=logit_bxe,
                metrics=['accuracy'])

    Trop.evaluate(bh_test_tuple)

    #Net.evaluate(bh_test_tuple)#98ish






    ##Why doesnt this work instead??##
#    @tf.function
#    def trop_rat_fun(image):
#        #ed_X=tf.expand_dims(image,1)
#        f_trop,g_trop=trop_polys(image,ed_Sigma,layers)
#        F_last=tf.reduce_max(f_trop[-1],axis=1)
#        G_last=tf.reduce_max(g_trop[-1],axis=1)
#        return F_last - G_last
#
#    @tf.function
#    def test_fun(I):
#        return 2*I
#
#    trfX=trop_rat_fun(X)
##    Trop=keras.Model(inputs=[Input],outputs=trop_rat_fun(ed_Input) )
#    #Trop=keras.Model(inputs=[Input],outputs=trop_rat_fun(Input) )
#    Trop=keras.Model(inputs=[Input],outputs=test_fun(Input) )
#
#    trfX=Trop(X)
#    netX=Net(X)
#    print('diff:',DIFF(TropRat(X),Net(X)) )
#
##    Fsig,Gsig=TropPolys(Net.input)
#
#
#    ###Evaluate###
#    train_tuple,test_tuple=tuple_splits(datasets)
#    #bh_train_tuple=train_tuple.batch(bhsz).repeat()
#    bh_test_tuple = test_tuple.batch(bhsz)

    #Net.evaluate(bh_test_tuple)


#    ed_Sigma=[tf.expand_dims(sig,0) for sig in Sigma]
#    ed_input=tf.expand_dims(Net.input,1)


#    TropPolys = keras.Model(inputs=Net.inputs,
#            outputs=[poly[-1] for poly in trop_polys(ed_input,ed_Sigma,layers)]  )
#    Fsig,Gsig=TropPolys(Net.input)
#    F=tf.reduce_max(Fsig,axis=1)
#    G=tf.reduce_max(Gsig,axis=1)
#    TropRat = keras.Model(inputs=Net.inputs,
#            outputs=F - G )




    #f_trop,g_trop=trop_polys(X,lrn_Sigma,layers)

#    #f_trop,g_trop=trop_polys(X,Sigma,layers)
#    trop_lin=[ff-gg for ff,gg in zip(f_trop,g_trop)]
#
#    sig_act,sig_lin=net_at_state_map(X,lrn_Sigma,layers,True)
#    net_act=Acts(X)
#
#
#    ed_Sigma=[tf.expand_dims(sig,0) for sig in Sigma]
#    ed_input=tf.expand_dims(Net.input,1)
#    TropPolys = keras.Model(inputs=Net.inputs,
#            outputs=[poly[-1] for poly in trop_polys(ed_input,ed_Sigma,layers)]  )
#    Fsig,Gsig=TropPolys(Net.input)
#    F=tf.reduce_max(Fsig,axis=1)
#    G=tf.reduce_max(Gsig,axis=1)
#    TropRat = keras.Model(inputs=Net.inputs,
#            outputs=F - G )







#scratch#

#tried to get unique states, but not helpful
#    print('get unique Sigma')
#    print('of course.. theyre all unique')
#    # def get_net_states
#    hidden_states=np.concatenate(np_State[:-1],axis=1)
#    Sig,Inv,Cnts=np.unique(hidden_states,axis=0,
#                           #return_index=True,
#                           return_inverse=True,
#                           return_counts=True)
#
#    #Idx0 might not be meaningful if not gridX
#    IdxPlus =Inv[np.where(hidden_states==1)[0]]
#    IdxMinus=Inv[np.where(hidden_states==0)[0]]
#    Idx0=np.intersect1d(IdxPlus,IdxMinus)
#
#    Centers=np.zeros([len(Sig)]+np_X.shape[1:])
#    #for k in range(len(Sig)):#region
#    for k in np.unique(Inv): #equivalent
#        Xk=np_X[np.where(Inv==k)[0]]#the corresp x's
#        Centers[k]=np.mean(Xk,axis=0)
#    Sig0=Sig[Idx0]
#


