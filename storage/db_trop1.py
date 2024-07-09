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

from standard_data import *

from state_maps import pos_cmpt,neg_cmpt,abs_cmpt,pos_cmpts,neg_cmpts,abs_cmpts
from state_maps import (apply_linear_layer,apply_sig,apply_keras_layer,
                        fg_recurse,net_at_state_map,trop_polys
                       )

from utils import SH,DIFF,EQUAL #db handy

'''
Getting started with tensorflow2.0
'''
"""
mental notes:
"""




if __name__=='__main__':

    #--------#
    #stophere#
    #--------#

    config,_=get_config()
    prepare_dirs_and_logger(config)
    save_config(config)

    #print('model_dir:',config.model_dir)
    model_dir=config.model_dir
    #data_fn=get_toy_data(config.dataset)
    #experiment=Experiment(config,data_fn)
    #return experiment

    #model_name=os.path.join(checkpoint_dir,'Model')
    record_dir=os.path.join(config.model_dir,'records')
    checkpoint_dir=os.path.join(model_dir,'checkpoints')
    model_file=os.path.join(checkpoint_dir,'Model_ckpt.h5')
    summary_dir=os.path.join(model_dir,'summaries')
    make_folders([checkpoint_dir])#,summary_dir])
    print('[*] Model File:',model_file)


    ###--------End Main--Load Data----###

    #Prep Data
    datasets,info=load_frogs_vs_ships()
    #datasets,info=load_mnist()

    train_data,test_data=tuple_splits(datasets)

    #---debug datasets---#
    db_bat=peak(datasets['train'].batch(20))
    X=db_bat['x']
    db_bat25=peak(datasets['train'].batch(25))
    X25=db_bat25['x']
    X20=X

    ds_train=datasets['train']
    ds_input=ds_train.map(lambda e:e['x']).batch(1)
    #input_shape=[None]+db_bat['x'].shape[1:]
    input_shape=db_bat['x'].shape[1:]
    #input_shape=[None]+tf.compat.v1.data.get_output_shapes(ds_input)
    #input_shape=[None]+ds_input.output_shapes[1:]
    #ds_input.output_shapes
    #---debug datasets---#

    train_size=info.splits['train'].num_examples
    test_size=info.splits['test'].num_examples
    buf=np.int(train_size*0.1)

    #Yes. was causing stall.hm
    #train_data=train_data.shuffle(buf)#was this causing stall?

    train_data=train_data.batch(config.batch_size).repeat()
    test_data=test_data.batch(config.batch_size).repeat()

    #stophere


#-------------Begin Model--------------#


    if config.load_path:
        new_model=keras.models.load_model(config.load_path)
        stophere

    ##Code relies on terminating with linear and using bxe(with_logits=True)

    #Model2   #also works fine on mnist
    model=keras.Sequential([
        keras.layers.Conv2D(64,3,activation='relu',input_shape=input_shape),
        #keras.layers.MaxPool2D( (2,2) ),#recently commented
        ##keras.layers.Conv2D(128,3,2,activation='relu'),
        keras.layers.Conv2D(128,3,activation='relu'),
        #keras.layers.MaxPool2D( (2,2) ),#recently commented
        keras.layers.Conv2D(64,3,activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(64,activation='relu'), #-1 if commented out

        keras.layers.Dense(1), #XE from logits
        #keras.layers.Dense(1,activation='sigmoid'),
    ])


    #'./logs/Model_0803_154817_frogship_debug'

    #model.build(input_shape=input_shape)

    L0,L1,L2,L3,L4=model.layers[:5]
    Llast=model.layers[-1]

    #lr=0.01 is what the mnist experiments were done on
    #lr=0.005 works for shallower cifar models

#    model.compile(
#                #optimizer='adam',
#                optimizer=opt,
#                #loss='sparse_categorical_crossentropy',
#                loss='binary_crossentropy',#can we change to sigmoid xpy
#                metrics=['accuracy'])


    def lyr2state(layer):
        #will need to modify to do maxpool
        #cleanly handle sigmoid terminal layer #though prefer logit=model.output
        tare = layer.activation(0.) if hasattr(layer,'activation') else 0.
        #Yes, this will produce a state for "flatten" layers, but to no effect
        return tf.cast( tf.greater(layer.output, tare), tf.float32 )

    sig_list=[lyr2state(L) for L in model.layers]
    State=keras.Model(inputs=model.inputs,outputs=sig_list)
    Acts=keras.Model(inputs=model.inputs,
         outputs=model.inputs+[L.output for L in model.layers])#InputAreActs


    print('DEBUG: learning keras.Model')
    #Linear0=keras.Model(inputs=model.inputs,output=apply_keras_layer(L0.input,L0))
    #Linear0=keras.Model(inputs=model.inputs,outputs=apply_keras_layer(model.input,L0))#works
         #outputs=[L.output for L in model.layers])

    #L0_model=keras.Model(model.inputs,L0(model.input))

    Sigma20=State(X20)
    Sigma25=State(X25)

#############################
    ## code dev config ##
    #Sigma=Sigma20
    Sigma=Sigma25
    layers=model.layers
    X=X20
#############################


    sig_act,sig_lin=net_at_state_map(X,Sigma,layers,True)
    net_act=Acts(X)

    if Sigma[0].shape[0]==X.shape[0]:
        print('net_at_state( sig(x)) is net(x):')
        same_act=[EQUAL(L,N) for L,N in zip(sig_act,net_act)]
        for ll,sa in enumerate(same_act):
            print(ll,sa)
### (end) net_state_map ###


    f_trop,g_trop=trop_polys(X,Sigma,layers)

    #can make Sigma another input
#    TropRat=keras.Model(inputs=model.inputs,
#            outputs=[tf.reduce_max(trop_polys(model.input,Sigma,layers)[0],axis=1) - \
#                     tf.reduce_max(trop_polys(model.input,Sigma,layers)[1],axis=1)]
#                         )

    trop_lin=[ff-gg for ff,gg in zip(f_trop,g_trop)]

    f1,f2,f3,f4,f5,f6,f7=f_trop
    g1,g2,g3,g4,g5,g6,g7=g_trop
    #p1,p2,p3,p4,p5,p6,p7=trop_lin#not quite right
    tl1,tl2,tl3,tl4,tl5,tl6,tl7=trop_lin
    sl1,sl2,sl3,sl4,sl5,sl6,sl7= sig_lin
    sig1,sig2,sig3,sig4,sig5,sig6,sig7=Sigma

    #some numerical issues
    trop_diff_lin=[DIFF(T,S) for T,S in zip(trop_lin,sig_lin) ]
    print('Layerwise Numerical Trop Error:')
    for ll,dl in enumerate(trop_diff_lin):
        print(ll,dl)


    #if f_trop[-1].ndim==3:#batch x sig x 1

    assert( f_trop[-1].ndim==3 )#batch x sig x 1



    ed_Sigma=[tf.expand_dims(sig,0) for sig in Sigma]
    ed_input=tf.expand_dims(model.input,1)
    TropPolys = keras.Model(inputs=model.inputs,
            outputs=[poly[-1] for poly in trop_polys(ed_input,ed_Sigma,layers)]  )
    Fsig,Gsig=TropPolys(model.input)
    F=tf.reduce_max(Fsig,axis=1)
    G=tf.reduce_max(Gsig,axis=1)
    TropRat = keras.Model(inputs=model.inputs,
            outputs=F - G )
    #  %timeit -n 10  tout=TropRat(X)  #135ms
    #  %timeit -n 100  out=    Net(X)  #5ms

    #TropRat(X)==Net(X) #True






    #TropRat( X ) # takes awhile but works
    ##OKAY whew. I got tropical poly to work##




#    print('DEBUGZONE')#~~- Eventually fixed numerical issue
#    #starting point
#    print(' SMALL:|rs(sig3*tl3)-sl4|=',DIFF( tf.reshape(sig3*tl3,[20,-1]),sl4 ))
#
#    db_f,db_g=f3,g3
#    db_sig=sig3
#    db_shape=[-1,20]
#    db_lyr=L3
#    #------------------
#
#    db_p=apply_sig(db_f-db_g,db_sig)#DIFF sl4 still small!
#
#    pos_wts=pos_cmpts(db_lyr.weights)
#    neg_wts=neg_cmpts(db_lyr.weights)
#    abs_wts=abs_cmpts(db_lyr.weights)
#
#    #ord_lin=apply_keras_layer(p,db_lyr)
#    db_pos_lin=apply_keras_layer(db_p,db_lyr,with_weights=pos_wts)#diff sl4 SMALL
#    db_neg_lin=apply_keras_layer(db_p,db_lyr,with_weights=neg_wts)
#    db_abs_lin=apply_keras_layer(db_g,db_lyr,with_weights=abs_wts)#abs(bias) will cancel in f-g
#
#    db_f_next=db_pos_lin+db_abs_lin
#    db_g_next=db_neg_lin+db_abs_lin
#
#    db_f4=db_f_next
#    db_g4=db_g_next



#    #trop_same_lin=[tf.reduce_all(T==N) for T,N in zip(trop_lin,sig_lin)]
#    trop_same_lin=[DIFF(T,S)<=1e-4 for T,S in zip(trop_lin,sig_lin)]
#    print('\nVerify trop_rat')
#    for ll,sa in enumerate(trop_same_lin):
#        print(ll,sa)

    #tl2 /not= sl2  #trop_linear vs sig_linear

    ###Have sig_act, net_act,and trop_act. All have to be equal###


#    #DEBUG#  is nn.conv2d linear?
#    #lyr=layers[1]
#    #ker1=lyr.weights[0]
#    ker1=L1.weights[0]
#    p1=apply_sig(f1-g1,sig1)
#    f,g,p=f1,g1,p1
#
#    ker_pos,ker_neg=pos_cmpt(ker1),neg_cmpt(ker1)
#    args=[L1.strides,L1.padding.upper(),]
#    conv_ord=tf.nn.conv2d(p,ker1,*args)
#    conv_pos=tf.nn.conv2d(p,ker_pos,*args)
#    conv_neg=tf.nn.conv2d(p,ker_neg,*args)
#
#    #numerical inexactness
#    tf.reduce_all( tf.abs( conv_ord-(conv_pos-conv_neg))<=1e-5 )#True

    #conv_pos = tf.nn.conv2d(p, weights[0], layer.strides,
    #                layer.padding.upper()) + weights[1]



    ##!!!get_sig won't work on sigmoid output# See inspect_saved
    #State.weights #still works

    #sigL0=State(X)[0]
    #sig=sigL0


    #input_idx,input_shape=X.shape[:ix_features]  ,X.shape[ix_features:]
    #sig_idx,sig_shape    =sig.shape[:ix_features],sig.shape[ix_features:]


#    sig20=State(X20)[0]
#    sig25=State(X25)[0]
#
#    lin20=apply_linear_layer(X20,L0)
#    lin25=apply_linear_layer(X25,L0)
#
#    act20=apply_sig(lin20,sig20)
#    act_mix=apply_sig(lin20,sig25)
#
#    h1X20=L0(X20)
#    print(tf.reduce_all(  h1X20 == act20 ))

    ##


    #def fwdpass_at_state(x, sig, layer=None,weights=None):


#    h1X=L0(X)
#    ALL1X= tf.nn.relu( apply_linear_layer(X,L0) )
#    ALL1Xw= tf.nn.relu( apply_linear_layer(X,L0,with_weights=L0.weights) )
#    print( tf.reduce_all( h1X==ALL1X) )#works
#    print( tf.reduce_all( h1X==ALL1Xw) )#works


    #def fwdpass_at_state(x, sig, layer):


#    tf.nn.relu(tf.nn.conv2d(X,L0.weights[0], L0.strides,
#                            L0.padding.upper()) + L0.weights[1])
#    if isinstance( L0, keras.layers.Conv2D ):
#        tf.nn.relu(tf.nn.conv2d(X,L0.weights[0],L0.strides,L0.padding.upper())+L0.weights[1])



    #these two are the same
    #h1X=L0(X)
    #nn1X=tf.nn.relu(tf.nn.conv2d(X,L0.weights[0],L0.strides,L0.padding.upper())+L0.weights[1])
    #tf.reduce_all( h1X==nn1X )#returns True


    ##Write things out functionally##

    ##Inputs : SigList, X;
    ##Outputs: Y

    ##model/weights+sigma->tropf

    #trop_model=keras.Model(inputs=(model.inputs,sigl),
    #                       outputs=ftrop,gtrop


    ## Tried to extend existing model. no good. ##
#    Lnew=keras.layers.Dense(units=L4.units)
#    Lnew.build(L4.input_shape)
#    Lnew.set_weights(L4.get_weights())
#    Wnew=keras.activations.relu(  L4.weights[0]  )
#
#    @tf.function
#    def get_wnew():
#        return tf.nn.relu( L4.weights[0] )
#
#
#    print('before training')
#    print('L4:',L4.weights[0][0:2,0:5]    )
#    #print('Lnew:',Lnew.weights[0][0:2,0:5])
#    #print('wnew2:',Wnew[0:2,0:5]          )
#    print('wnew3:',get_wnew()[0:2,0:5]          )
#
#    #sig1=tf.cast( tf.greater(L1.output,0.), tf.float32 )
#    #stophere
#
#
#    short_td=train_data.take(12)#just for debug
#    model.fit(short_td)
#
#    print('after training')
#    print('L4:',L4.weights[0][0:2,0:5]    )
#    #print('Lnew:',Lnew.weights[0][0:2,0:5])
#    #print('wnew2:',Wnew[0:2,0:5]          )
#    print('wnew3:',get_wnew()[0:2,0:5]          )
#    ###Neither worked!   damn eager execution



    opt=keras.optimizers.Adam(lr=0.005)#(lr=0.01)
    bxe_from_logits=tf.losses.BinaryCrossentropy(from_logits=True)
    model.compile(
                #optimizer='adam',
                optimizer=opt,
                #loss='sparse_categorical_crossentropy',
                #loss='binary_crossentropy',
                loss=bxe_from_logits,
                metrics=['accuracy'])


    #--------#
    #stophere#(b4 train loop)
    #--------#

    ### A proper training and save loop ###
    spe=np.int(train_size/config.batch_size)
    val_steps=np.int(test_size/config.batch_size)
    if config.is_train:
        model.fit(train_data,
                  epochs=20,
                  steps_per_epoch=spe,  #1563,
                  validation_data=test_data,
                  validation_freq=2,#how many train epoch between
                  validation_steps=val_steps,
                 )
        db_pred=model.predict(db_bat['x'])
        model.save(model_file)

    print('log_dir=',model_dir)
    #stophere



#-----------------
#tricks and useful code
    #Can use .batch(12000).take(1) to bring into memory

#    #debug
#    db_train_data=train_data.batch(250).take(1)
#    db_data=next(iter(db_train_data))


    #model.fit(train_data.batch(32))



    #model.fit(train_data.batch(32))
        #simply call again to continue training

    ##Useful!
    #model.get_config()
    #met=model.metrics[0]
        #met.count.numpy()




    #rz_data=train_data.map(preprocess_images)
    #rz_datum=next(iter(rz_data))
    #rz_img=rz_datum['image'].numpy()
    ##rz_img=next(iter(rz_data))

    #plt.imshow(rz_img.astype(np.uint8))
    #plt.show()

#    #mnist model  #works on mnist
#    model=keras.Sequential([
#        keras.layers.Conv2D(32,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.Flatten(),
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(1,activation='sigmoid'),
#    ])



#Ex: fit()
#callbacks = [
#  # Write TensorBoard logs to `./logs` directory
#  keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
#]
#model.fit(train_dataset, epochs=100, steps_per_epoch=1500,
#          validation_data=valid_dataset,
#          validation_steps=3, callbacks=callbacks)

#Example
#    ##mnist arch
#    model=keras.Sequential([
#        keras.layers.Conv2D(32,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.Flatten(),
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(1,activation='sigmoid'),
#    ])

#    #Example
#    #https://www.tensorflow.org/beta/tutorials/keras/basic_classification
#    fashion_mnist = keras.datasets.fashion_mnist
#
#    (train_images, train_labels), (test_images, test_labels) =\
#        fashion_mnist.load_data()
#
#    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle, boot']
#
#    train_images = train_images / 255.0
#    test_images = test_images / 255.0
#
#
#    model = keras.Sequential([
#            keras.layers.Flatten(input_shape=(28, 28)),
#            keras.layers.Dense(128, activation='relu'),
#            keras.layers.Dense(10, activation='softmax'),
#        ])
#
#
#    model.compile(optimizer='adam',
#                  loss='sparse_categorical_crossentropy',
#                  metrics=['accuracy'])
#
#    model.fit(train_images, train_labels, epochs=10)
#    test_loss, test_acc = model.evaluate(test_images, test_labels)
#    print('\nTest accuracy:', test_acc)
#
#    predictions = model.predict(test_images)
#
#    np.argmax(predictions[0])
#    test_labels[0]


#Example:
#https://adventuresinmachinelearning.com/keras-eager-and-tensorflow-2-0-a-new-tf-paradigm/
#class CIFAR10Model(keras.Model):
#    def __init__(self):
#        super(CIFAR10Model, self).__init__(name='cifar_cnn')
#        self.conv1 = keras.layers.Conv2D(64, 5,
#                                         padding='same',
#                                         activation=tf.nn.relu,
#                                         kernel_initializer=tf.initializers.variance_scaling,
#                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
#        self.max_pool2d = keras.layers.MaxPooling2D((3, 3), (2, 2), padding='same')
#        self.max_norm = keras.layers.BatchNormalization()
#        self.conv2 = keras.layers.Conv2D(64, 5,
#                                         padding='same',
#                                         activation=tf.nn.relu,
#                                         kernel_initializer=tf.initializers.variance_scaling,
#                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
#        self.flatten = keras.layers.Flatten()
#        self.fc1 = keras.layers.Dense(750, activation=tf.nn.relu,
#                                      kernel_initializer=tf.initializers.variance_scaling,
#                                      kernel_regularizer=keras.regularizers.l2(l=0.001))
#        self.dropout = keras.layers.Dropout(0.5)
#        self.fc2 = keras.layers.Dense(10)
#        self.softmax = keras.layers.Softmax()
#
#    def call(self, x):
#        x = self.max_pool2d(self.conv1(x))
#        x = self.max_norm(x)
#        x = self.max_pool2d(self.conv2(x))
#        x = self.max_norm(x)
#        x = self.flatten(x)
#        x = self.dropout(self.fc1(x))
#        x = self.fc2(x)
#        return self.softmax(x)

##Example:
##https://adventuresinmachinelearning.com/keras-eager-and-tensorflow-2-0-a-new-tf-paradigm/
#class CIFAR10Model(keras.Model):
#    def __init__(self):
#        super(CIFAR10Model, self).__init__(name='cifar_cnn')
#        self.conv1 = keras.layers.Conv2D(64, 5,
#                                         padding='same',
#                                         activation=tf.nn.relu,
#                                         kernel_initializer=tf.initializers.VarianceScaling,
#                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
#        self.max_pool2d = keras.layers.MaxPooling2D((3, 3), (2, 2), padding='same')
#        self.max_norm = keras.layers.BatchNormalization()
#        self.conv2 = keras.layers.Conv2D(64, 5,
#                                         padding='same',
#                                         activation=tf.nn.relu,
#                                         kernel_initializer=tf.initializers.VarianceScaling,
#                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
#        self.flatten = keras.layers.Flatten()
#        self.fc1 = keras.layers.Dense(750, activation=tf.nn.relu,
#                                      kernel_initializer=tf.initializers.VarianceScaling,
#                                      kernel_regularizer=keras.regularizers.l2(l=0.001))
#        self.dropout = keras.layers.Dropout(0.5)
#        self.fc2 = keras.layers.Dense(10)
#        self.softmax = keras.layers.Softmax()
#
#    def call(self, x):
#        x = self.max_pool2d(self.conv1(x))
#        x = self.max_norm(x)
#        x = self.max_pool2d(self.conv2(x))
#        x = self.max_norm(x)
#        x = self.flatten(x)
#        x = self.dropout(self.fc1(x))
#        x = self.fc2(x)
#        return self.softmax(x)




#    input_shape=X.shape[ix_features:]#for conv2d
#    input_idx=X.shape[:ix_features]
#    sig_shape=sig.shape[ix_features:]
#    sig_idx=sig.shape[:ix_features:]

