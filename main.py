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



#from state_maps import pos_cmpt,neg_cmpt,abs_cmpt,pos_cmpts,neg_cmpts,abs_cmpts
#from state_maps import (apply_linear_layer,apply_sig,apply_keras_layer,
#                        fg_recurse,net_at_state_map,trop_polys
#                       )

#from v1_tropical_layers import (TropicalMaxPool2D, TropicalReLU,
#                             TropicalFlatten, TropicalEmbed,
#                             TropicalConv2D, TropicalDense,
#                             tropical_objects,tropical_equivalents,


'''
ordinary "vanilla" deep learning code

This file is for training and retraining of tf NN models
with backprop.

Use this to generate/save standard models to experiment with later
'''


'''
#cmmn abbrievs#
db  debug
bh  batch
sz  size
rs  reshape
ed  expand_dims
ds  tf.Dataset
np  numpy
pm  plus or minus
'''



if __name__=='__main__':

    ### File/Folder Creation/Handling ###

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
    datasets,info=getattr(standard_data,config.data)()
    #data_get=getattr(standard_data,config.data)
    #datasets,info=data_get()

    ds_train=datasets['train']
    ds_input=ds_train.map(lambda e:e['x']).batch(1)

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



#-------------Begin Model--------------#

    if config.load_path:
        #loads old model for new experiments
        #old model_dir/ folder will be unchanged

        #model=keras.models.load_model(config.load_path)
        #model=keras.models.load_model(config.load_path,
        model=keras.models.load_model(config.load_model_file,#pass h5 file
            custom_objects=tropical_objects)

        #maybe save model in new folder?
    else:
        #New model
        model_get=getattr(standard_model,config.model)
        model=model_get(input_shape=info.input_shape)

    #ref
    db_logit=model(X)


#-------------Begin Train--------------#
    #opt=keras.optimizers.Adam(lr=0.005)#(lr=0.01)
    opt=keras.optimizers.Adam(lr=config.learning_rate)#0.005 default in config
    model.compile(
                optimizer=opt,
                loss=logit_bxe,
                metrics=['accuracy'])
    db_pred=model.predict(X) #db


    if config.is_train:

        ###Prep Data###
        train_tuple,test_tuple=tuple_splits(datasets)
        bh_train_data=train_tuple.batch(bhsz).repeat()
        bh_test_data = test_tuple.batch(bhsz).repeat()
        #   could do shuffle data if it were a big deal


        #train_steps=np.int(train_size/config.batch_size)
        #val_steps=np.int(test_size/config.batch_size)
        train_steps=np.int(train_size/bhsz)
        val_steps  =np.int(test_size/bhsz)
        history = model.fit(bh_train_data,
                  epochs=config.epochs,
                  steps_per_epoch=train_steps,
                  validation_data=bh_test_data,
                  validation_freq=2,#how many train epoch between
                  validation_steps=val_steps,
                 )
        model.save(model_file)
        print('saving at log_dir=',model_file)
    #stophere

    #Other saving
#json_config = model.to_json()
#reinitialized_model = keras.models.model_from_json(json_config)
#
#weights = model.get_weights()  # Retrieves the state of the model.
#model.set_weights(weights)  # Sets the state of the model.
#
#config = model.get_config()
#weights = model.get_weights()
#
#new_model = keras.Model.from_config(config)
#new_model.set_weights(weights)
#
## Check that the state is preserved
#new_predictions = new_model.predict(x_test)
#np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)
##https://www.tensorflow.org/guide/keras/save_and_serialize
#model.save_weights('path_to_my_tf_checkpoint', save_format='tf')



    #callback examples
#      model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#        'training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
#      early_stopping_checkpoint = keras.callbacks.EarlyStopping(patience=5)
#
#      history = model.fit(train.repeat(),
#          epochs=epochs,
#          steps_per_epoch=steps_per_epoch,
#          validation_data=validation.repeat(),
#          validation_steps=validation_steps,
#          callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping_checkpoint])
