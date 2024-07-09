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
from utils import prepare_dirs_and_logger,save_config,make_folders
from utils import logit_bxe #cross entropy helper
from utils import SH,DIFF,EQUAL #db handy
from config import get_config


#from standard_data import *
import standard_data
import standard_model
from standard_data import peak,inspect_dataset,tuple_splits
from standard_model import tropical_objects #help load custom layers


from tropical_layers import (TropicalMaxPool2D, TropicalReLU,
                             TropicalFlatten,# TropicalEmbed,
                             TropicalConv2D, TropicalDense,
                             tropical_objects,tropical_equivalents,
                             TropicalSequential,
                             wts2lyr
                            )


from oldstyle.vis_utils import (split_posneg , get_path, get_np_network,
                        get_neuron_values, splitL, load_weights,
                        resample_grid,vec_get_neuron_values,
                        get_del_weights, subplots)

from tboard import file2number

#from state_maps import pos_cmpt,neg_cmpt,abs_cmpt,pos_cmpts,neg_cmpts,abs_cmpts
#from state_maps import (apply_linear_layer,apply_sig,apply_keras_layer,
#                        fg_recurse,net_at_state_map,trop_polys
#                       )

#from v1_tropical_layers import (TropicalMaxPool2D, TropicalReLU,
#                             TropicalFlatten, TropicalEmbed,
#                             TropicalConv2D, TropicalDense,
#                             tropical_objects,tropical_equivalents,



'''
Some models with known numpy weights that I want to look at in particular

Have to format them as keras models
'''


if __name__=='__main__':


    print('WARNING v1_logdir hardcoded, be sure to switch it')
    #v1_logdir='./logs/Fig_Model_0505_005701_noise1D1A3'
    #v1_logdir='./logs/Fig_Model_0505_003620_noise2D1A3'
    #v1_logdir='./logs/Fig_Model_0505_012310_noise3D1A3'
    v1_logdir='./logs/Pub_Model_0504_233305_D1A3'

    #Setup
    print('using v1_logdir:',v1_logdir)
    record_dir=os.path.join(v1_logdir,'records')
    id_str=str(file2number(v1_logdir))
    Mm_dir=os.path.join(v1_logdir,'Mm_analysis')
    if not os.path.exists(Mm_dir):
        os.makedirs(Mm_dir)

    all_step=np.load(get_path('step','wwatch',v1_logdir))
    all_weights=load_weights(v1_logdir)#d*2*Txn1xn2
    del_weights=get_del_weights(all_weights)
    arch=[b.shape[-1] for w,b in del_weights[:-1]]#net architecture

    step=all_step[-1]
    weights=[[w[-1],b[-1]] for w,b in del_weights]#d*2*wtshape
    #Wweights,Bweights=zip(*weights)
    #time_weights=weights#backcompat

    gridX=np.load(get_path('gridX','hmwatch',v1_logdir))
    GridX=resample_grid(gridX)#200





#-------------Config and Log Handling--------------#
    config,_=get_config()#arg parsing
    prepare_dirs_and_logger(config)

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

    config.model_file=os.path.join(checkpoint_dir,'Model_ckpt.h5')
    model_file=config.model_file
    save_config(config)
    print('[*] Model File:',model_file)


#-------------Begin Data--------------#
    #datasets,info=standard_data.v1_noisy1_data1()
    datasets,info=getattr(standard_data,config.data)()

    ds_train=datasets['train']
    ds_input=ds_train.map(lambda e:e['x']).batch(1)

    train_size=info.train_size
    test_size=info.test_size

    #---reference batch for datasets---#
    db_bat=peak(ds_train.batch(50))
    X=db_bat['x']
    Y=db_bat['y']
    X20=X[:20]


#-------------Begin Model--------------#
    linear_layers=[wts2lyr(wb) for wb in weights]
    layers=[linear_layers[0]]
    for L in linear_layers[1:]:
        layers.append(TropicalReLU())
        layers.append(L)
    model=TropicalSequential( layers )


#    if config.load_path:
#        #loads old model for new experiments
#        #old model_dir/ folder will be unchanged
#
#        #model=keras.models.load_model(config.load_path)
#        #model=keras.models.load_model(config.load_path,
#        model=keras.models.load_model(config.load_model_file,#pass h5 file
#            custom_objects=tropical_objects)
#
#        #maybe save model in new folder?
#    else:
#        #New model
#        model_get=getattr(standard_model,config.model)
#        model=model_get(input_shape=info.input_shape)
#
#    #ref
    db_logit=model(X)



##-------------Begin Train--------------#
#    #opt=keras.optimizers.Adam(lr=0.005)#(lr=0.01)
    opt=keras.optimizers.Adam(lr=config.learning_rate)#0.005 i think default
    model.compile(
                optimizer=opt,
                loss=logit_bxe,
                metrics=['accuracy'])
    db_pred=model.predict(X) #db


    ###Prep Data###
    train_tuple,test_tuple=tuple_splits(datasets)
    #bh_train_data=train_tuple.batch(bhsz).repeat()
    #bh_test_data = test_tuple.batch(bhsz).repeat()
    bh_train_data=train_tuple.batch(bhsz)
    bh_test_data = test_tuple.batch(bhsz)


    model.evaluate(bh_train_data)
    #model.evaluate(bh_test_data)


#        #train_steps=np.int(train_size/config.batch_size)
#        #val_steps=np.int(test_size/config.batch_size)
#        train_steps=np.int(train_size/bhsz)
#        val_steps  =np.int(test_size/bhsz)
#        history = model.fit(bh_train_data,
#                  epochs=config.epochs,
#                  steps_per_epoch=train_steps,
#                  validation_data=bh_test_data,
#                  validation_freq=2,#how many train epoch between
#                  validation_steps=val_steps,
#                 )

    model.save(model_file)
    print('saving at log_dir=',model_file)

    reload_model= keras.models.load_model(model_file,#pass h5 file
            custom_objects=tropical_objects)
    print('successful reload')
    #stophere


