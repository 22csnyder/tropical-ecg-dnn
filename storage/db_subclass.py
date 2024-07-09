from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#from keras import backend as K
#from keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import layers

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

from standard_data import *
#from standard_data import peak,inspect_dataset,tuple_splits
from utils import logit_bxe

from state_maps import pos_cmpt,neg_cmpt,abs_cmpt,pos_cmpts,neg_cmpts,abs_cmpts
from state_maps import (apply_linear_layer,apply_sig,apply_keras_layer,
                        fg_recurse,net_at_state_map,trop_polys
                       )


##For SignDense and SignConv2D, I want them to do exactly the same thing,
#but on call, be able to specify which combo of kernel+ or kernel- to use


relu=tf.nn.relu

def sgn_split(ten):
    return map(relu,[ten,-ten])

def pm_cmpt(ten,pm):
    if pm=='lin':
        return ten
    else:
        tpos,tneg=sgn_split(ten)
        tabs=tpos+tneg

        if pm=='pos':
            return tpos
        elif pm=='neg':
            return tneg
        elif pm=='abs':
            return tabs
        else:
            raise ValueError('arg pm,',pm,' ,wasnt in lin,pos,neg,abs')


#def pm_splits(weights,pm):#list arg
#    '''
#    weights should be list
#    '''
#    if pm=='lin':
#        return weights
#    else:
#        wsplit=[map(relu,[w,-w]) for w in weights]
#        wpos,wneg=zip(*wsplit)
#        wabs=[wp+wn for wp,wn in wsplit]
#
#        if pm=='pos':
#            return wpos
#        elif pm=='neg':
#            return wneg
#        elif pm=='abs':
#            return wabs
#        else:
#            raise ValueError('arg pm,',pm,' ,wasnt in lin,pos,neg,abs')



##For SignDense and SignConv2D, I want them to do exactly the same thing,
#but on call, be able to specify which combo of kernel+ or kernel- to use

class SignWrap(layers.Layer):
    legal_pm=['lin','abs','pos','neg']#intended read-only
    @property
    def pm(self):
        #print('SignWrap.pm')
        if not hasattr(self,'_pm'):
            self.pm_reset()
        return self._pm
    @pm.setter
    def pm(self,val):
        #print('SignWrap.pm.setter')
        if not val in self.legal_pm:
            raise ValueError('Unexpected pm arg. Expected one of:',
                             self.legal_pm,'but got instead',val)
        self._pm=val
    def pm_reset(self):
        self.pm='lin'

    @property
    def kernel(self):
        #print('signwrap.kernel ')
        return pm_cmpt(self._kernel,self.pm)
    @kernel.setter
    def kernel(self,ten):
        #print('signwrap.kernel.setter ')
        self._kernel=ten

    @property
    def bias(self):
        #print('signwrap.bias ')
        return pm_cmpt(self._bias,self.pm)
    @bias.setter
    def bias(self,ten):
        #print('signwrap.bias.setter ')
        self._bias=ten


    def __call__(self, inputs, pm='lin'):
        #Unusually here, you actually need to wrap __call__() instead of call()
        #"do what you would have before but with
        #  the specified version of self.kernel and self.bias"

        #print('__signwrap __call__')


        #Layer.__call__() now will
        #1)set pm  ('abs' for example)
        #2)build if not built
        #  2.5)set kernel
        #3)get kernel
        #  3.5)get pm (to get kernel split)
        #3)reset pm to 'lin'

        self.pm=pm# affects self.kernel getter method
        #out=super(SignWrap,self).__call__(self,inputs)   #doesnt
        out=layers.Layer.__call__(self,inputs)            #works
        self.pm_reset()
        return out

    def equiv_layer(self,inputs):
        #Simply the linear layer without worry about sign components
        self.pm_reset()
        self.__call__(self,inputs)



#class SignDense(layers.Dense,SignWrap):
class SignDense(SignWrap,layers.Dense):
    def __init__(self,*args,**kwargs):
        ##Insist no nonlinearity. Otherwise, pm methods make no sense
        kwargs['activation']=None
        super(SignDense,self).__init__(*args,**kwargs)

#class SignConv2D(layers.Dense,SignWrap):
class SignConv2D(SignWrap,layers.Conv2D):
    def __init__(self,*args,**kwargs):
        ##Insist no nonlinearity. Otherwise, pm methods make no sense
        kwargs['activation']=None
        super(SignConv2D,self).__init__(*args,**kwargs)

#SignWrap,Conv2D,and Dense all inherit from Layer
#  but no conflict in method overrides
#multi inherit order doesnt matter because Dense,Conv2D dont override Layer.__call__
#both use "self.kernel" and "self.bias" to access weights
#these get overridden by properties/methods kernel(),bias() of SignWrap



#Tropical versions of regular layers
#Computes the effect on the pair [f,g]
#  [f2, g2] = L( [f,g] ) 
#instead of on x, if it were the case that x=f-g
#all implement self.equiv_layer, so that 
#  f2,g2=L.equiv_layer(f-g)


##Can also do fancy wrapping things instead
#keras.engine.base_layer.wrapped_fn()
#keras.layers.TimeDistributed(layer)
#class SecondTropicalDense(layers.Layer):
class TropicalDense(layers.Layer):
    def __init__(self,*args,**kwargs):
        super(TropicalDense,self).__init__()
        self.dense=SignDense(*args,**kwargs)
        #self.equiv_layer=self.dense.equiv_layer
        self.equiv_layer=self.dense#eh
    def call(self,fg_pair):
        #print('TropDense.call')
        f,g=fg_pair
        f2=self.dense(f,pm='pos') + self.dense(g,pm='neg')
        g2=self.dense(g,pm='pos') + self.dense(f,pm='neg')
        if self.dense.use_bias:
            f2+=self.dense.bias
        return [f2,g2]
    def get_config(self):
        return self.equiv_layer.get_config()

class TropicalConv2D(layers.Layer):
    def __init__(self,*args,**kwargs):
        super(TropicalConv2D,self).__init__()
        self.conv=SignConv2D(*args,**kwargs)
        #self.equiv_layer=self.conv.equiv_layer
        self.equiv_layer=self.conv
    def call(self,fg_pair):
        #print('TropConv2D.call')
        f,g=fg_pair
        f2=self.conv(f,pm='pos') + self.conv(g,pm='neg')
        g2=self.conv(g,pm='pos') + self.conv(f,pm='neg')
        if self.conv.use_bias:
            f2+=self.conv.bias
        return [f2,g2]
    def get_config(self):
        return self.equiv_layer.get_config()

class TropicalEmbed(layers.Layer):
    '''
    Layer must start model
    Embeds input x as difference [x,0]
    '''
    def __init__(self,*args,**kwargs):
        #pass input_shape to parent
        super(TropicalEmbed,self).__init__(*args,**kwargs)
        #currently, kwargs is being fed to wrapped layer in other cases
        self.identity=layers.Lambda(lambda x:x)
        self.equiv_layer=self.identity
    def call(self,inputs):
        inputs=self.identity(inputs)
        return [inputs,tf.zeros_like(inputs)]
#    def compute_output_shape(self,input_shape):
#        #implement in case your layer modifies the shape of its input
#           #yet seems to work without
#        return [input_shape,input_shape]
#        #return input_shape,input_shape

class TropicalFlatten(layers.Layer):
    def __init__(self,*args,**kwargs):
        super(TropicalFlatten,self).__init__()
        self.flatten=layers.Flatten(*args,**kwargs)
        self.equiv_layer=self.flatten
    def call(self,fg_pair):
        f,g=fg_pair
        return map(self.flatten,[f,g])

class TropicalReLU(layers.Layer):
    def __init__(self,*args,**kwargs):
        super(TropicalReLU,self).__init__()
        self.relu=layers.ReLU(*args,**kwargs)
        self.equiv_layer=self.relu
    def call(self,fg_pair):
        #max(f,g)-g = relu(f-g)
        f,g=fg_pair
        return [tf.maximum(f,g),g]


class TropicalMaxPool2D(layers.Layer):
    def __init__(self,*args,**kwargs):
        super(TropicalMaxPool2D,self).__init__()
        kwargs['padding']='valid' #need constant patch size to sum_pool
        self.ave_pool=layers.AveragePooling2D(*args,**kwargs)
        self.max_pool=layers.MaxPooling2D(*args,**kwargs)#same as MaxPool2D
        self.n_pool  =np.prod(self.ave_pool.pool_size)
        self.equiv_layer=self.max_pool

    def call(self,fg_pair):
        f,g=fg_pair

        #sum pool
        G=self.n_pool * self.ave_pool( g )

        f2 = G + self.max_pool( f-g )
        g2 = G

        return [f2,g2]


tropical_objects={
    'TropicalMaxPool2D':TropicalMaxPool2D,
    'TropicalReLU'     :TropicalReLU,
    'TropicalFlatten'  :TropicalFlatten,
    'TropicalEmbed'    :TropicalEmbed,
    'TropicalConv2D'   :TropicalConv2D,
    'TropicalDense'    :TropicalDense,
}
tropical_equivalents={
    layers.MaxPooling2D:TropicalMaxPool2D,
    layers.ReLU        :TropicalReLU,
    layers.Flatten     :TropicalFlatten,
    #TropicalEmbed,
    layers.Conv2D      :TropicalConv2D,
    layers.Dense       :TropicalDense,
}

##T1,T2 same shape:  batchx(stuff)
#dotproduct over stuff to return either
#batchx1       <->  inner
#batchxbatchx1 <->  outer
def batch_inner(T1,T2):
    dots=tf.reduce_sum(T1*T2,axis=range(1,T1.ndim))
    return tf.expand_dims(dots,-1)
def batch_outer(T1,T2):
    edT1=tf.expand_dims(T1,1)
    edT2=tf.expand_dims(T2,0)
    allpairs_dots=tf.reduce_sum( edT1*edT2, axis=range(2,edT1.ndim) )
    return tf.expand_dims(allpairs_dots,-1)


if __name__=='__main__':
    print('db_subclass.py')

    #Should help with precision:
    #keras.backend.set_floatx('float64')

    ##aLoad Dataset###
    datasets,info=load_mnist()
    #datasets,info=load_fashion_mnist()
    #datasets,info=load_cifar10()
    train_data,test_data=tuple_splits(datasets)
    info_shape=info.features['image'].shape
    iis=info.input_shape
    input_shape=tf.TensorShape(iis)
    db_bat=peak(datasets['train'].batch(20))
    X=db_bat['x']
    O=tf.zeros_like(X)#looks like 0. is an Oh.
    l=tf.ones_like(X)#kinda looks like a 1
    db_bat25=peak(datasets['train'].batch(25))
    X25=db_bat25['x']
    X20=X
    ds_train=datasets['train']
    ds_input=ds_train.map(lambda e:e['x']).batch(1)
    input_shape=db_bat['x'].shape[1:]#equiv to tf.TS(info.is)
    ###Load Dataset###


    #tropmaxpool=TropicalMaxPool2D()
    #f_mp,g_mp=tropmaxpool([X,O])
    #f_tmp,g_tmp=tropmaxpool([X+l,l])
    #print('tropmaxpool works?',DIFF(f_mp-g_mp,f_tmp-g_tmp))


    ksz=(3,3)#kernel size
    sds=(2,2)#strides

    inputNode =layers.Input(input_shape)
    tropEmb   =TropicalEmbed()
    tropConv2D=TropicalConv2D(16,ksz,sds,input_shape=input_shape)
    tropMP    =TropicalMaxPool2D()
    tropFlat  =TropicalFlatten()
    tropDense1=TropicalDense(8)
    tropReLU  =TropicalReLU()
    #tropDense2=TropicalDense(3)
    tropDense2=TropicalDense(1)

    trop_layers=[
        tropEmb   ,
        tropConv2D,
        tropMP    ,
        tropFlat  ,
        tropDense1,
        tropReLU  ,
        tropDense2,
    ]
    orig_layers=[L.equiv_layer for L in trop_layers]



#    ta1=tropEmb(inputNode)
#    ta2=tropConv2D(ta1)
#    ta3=tropMP(ta2)
#    ta4=tropFlat(ta3)
#    ta5=tropDense1(ta4)
#    ta6=tropReLU(ta5)
#    ta7=tropDense2(ta6)



    orig_act=inputNode
    trop_act=inputNode
    for tL,oL in zip(trop_layers,orig_layers):
        trop_act=tL(trop_act)
        orig_act=oL(orig_act)



    tropModel=keras.Model(inputs=inputNode,outputs=trop_act)
    origModel=keras.Model(inputs=inputNode,outputs=orig_act)
    f_model  =keras.Model(inputs=inputNode,outputs=trop_act[0])
    g_model  =keras.Model(inputs=inputNode,outputs=trop_act[1])

    fY,gY=tropModel(X)
    Y    =origModel(X)
    print('does trop work?',DIFF(fY-gY,Y)) #Thank goodness


    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        f_Y,g_Y=tropModel(X)
    f_lin=tape.gradient(f_Y,X)
    g_lin=tape.gradient(g_Y,X)
    del tape

    f_lin_X=tf.reshape(tf.reduce_sum(X*f_lin,axis=range(1,X.ndim)),f_Y.shape)
    f_bias=f_Y - f_lin_X

    g_lin_X=tf.reshape(tf.reduce_sum(X*g_lin,axis=range(1,X.ndim)),g_Y.shape)
    g_bias=g_Y - g_lin_X

    #Try apply affine
    #f_tropX=tf.reshape(tf.reduce_sum(X*f_lin,axis=range(1,X.ndim)),f_bias.shape)+f_bias
    #g_tropX=tf.reshape(tf.reduce_sum(X*g_lin,axis=range(1,X.ndim)),g_bias.shape)+g_bias

    f_tropX=batch_inner(X,f_lin)+f_bias
    g_tropX=batch_inner(X,g_lin)+g_bias
    print('pwlinear grad works?',DIFF( f_tropX - g_tropX , Y ) )


    f_table=batch_outer(X,f_lin) + f_bias
    g_table=batch_outer(X,g_lin) + g_bias

    f_am=np.argmax(f_table,axis=0)
    g_am=np.argmax(g_table,axis=0)
    print('argmax s(x) is at x?', EQUAL(f_am,g_am) )





    ###prep training###
    train_size=info.splits['train'].num_examples
    test_size=info.splits['test'].num_examples
    buf=np.int(train_size*0.1)

    #Yes. was causing stall.hm
    #train_data=train_data.shuffle(buf)#was this causing stall?

    train_data=train_data.batch(32).repeat()
    test_data=test_data.batch(32).repeat()
    ###prep training###




    ##None of this code is current fyi##
    #   #just for debug
    opt=keras.optimizers.Adam(lr=0.005)#(lr=0.01)
    bxe_from_logits=tf.losses.BinaryCrossentropy(from_logits=True)
    origModel.compile(
                optimizer=opt,
                loss=bxe_from_logits,
                metrics=['accuracy'])

    ### A proper training and save loop ###
    spe=np.int(train_size/32.)
    val_steps=np.int(test_size/32.)

    origModel.fit(train_data,
              epochs=1,
              steps_per_epoch=spe,  #1563,
              validation_data=test_data,
              validation_freq=2,#how many train epoch between
              validation_steps=val_steps,
             )
    db_pred=origModel.predict(db_bat['x'])


    f_Y2,g_Y2=tropModel(X)
    Y2     =origModel(X)
    print('still same after train?',DIFF(f_Y2-g_Y2,Y2) )
    #print('(after train) output diff:')
    #print( tf.concat([f_Y2-g_Y2,Y2],axis=-1) )


    print('saving...')
    m_file='tmp/mymodel.h5'
    tropModel.save(m_file)

    print('loading..')
    load_model=keras.models.load_model(m_file,
            custom_objects=tropical_objects)#compile=False

    ##Seems to work pretty well##



#    ###seems to work now after fix bias situation###
#    fgdif=lambda fg:fg[0]-fg[1]
#
#    ##
#    tCA=tropConv2D([X,tf.zeros_like(X)])
#    oCA=tropConv2D.equiv_layer(X)
#    print('conv dif:',DIFF(fgdif(tCA),oCA))
#
#
#    scv=tropConv2D.conv #signconv layer#
#    slin=tropDense1.equiv_layer
#
#    scv_diff=DIFF(scv(X,'pos')-scv(X,'neg'),scv(X) )
#    print('SignConv okay?',scv_diff)
#
#
#    ##
#
#    flX=tropFlat.equiv_layer(X).numpy()[:,:576]
#    lp=slin(flX,pm='pos')
#    ln=slin(flX,pm='neg')
#    ld=slin(flX)
#    print('SignLin okay?',DIFF(lp-ln,ld) )
#
#    #debug why diff
#    OM=origModel
#    TM=tropModel
#
#
#    oL0,oL1,oL2,oL3=OM.layers[:4]
#    tL0,tL1,tL2,tL3=TM.layers[:4]
#
#    oA=[L(X) for L in OM.layers[:4]]
#    tA=[L(X) for L in TM.layers[:4]]
#
#    for i,acts in enumerate(zip(oA,tA)):
#        o,t=acts
#        print(i,' : ', DIFF(fgdif(t),o) )




#    oActs=[L(X) for L in OM.layers]
#    tActs=[fgdif( L(X) ) for L in OM.layers]
#
#    for i in range(len(oActs)):
#        print(i,' : ', DIFF(tActs[i],oActs[i]) )
#

















#####scratch##
#class FirstTropicalDense(object):
#    def __init__(self,*args,**kwargs):
#        self.layer=SignDense(*args,**kwargs)
#    def __call__(self,fg_pair):
#        f,g=fg_pair
#        f2=self.layer(f,pm='pos') + self.layer(g,pm='neg')
#        g2=self.layer(g,pm='pos') + self.layer(f,pm='neg')
#        return [f2,g2]


#
#    troplin=CustomSignDense(7)
#    y      =troplin(x)
#    ylin   =troplin(e)
#    ypos   =troplin(e,pm='pos')
#    yabs   =troplin(e,pm='abs')
#
#
#    wrap=SignWrapDense(8)
#    wr_lin=wrap( x , pm='lin')
#    wr_pos=wrap( x , pm='pos')
#    wr_neg=wrap( x , pm='neg')
#
#
#    sgnden=SignDense(8,activation=relu)#multi inherit
#    sg_lin=sgnden( x , pm='lin')
#    sg_pos=sgnden( x , pm='pos')
#    sg_neg=sgnden( x , pm='neg')
#    print('sg works?:',DIFF(sg_lin, sg_pos-sg_neg))



###########################################################################
#
#
#
#class SignWrapDense(layers.Dense):
#    legal_pm=['lin','abs','pos','neg']#intended read-only
#    @property
#    def pm(self):
#        if not hasattr(self,'_pm'):
#            self.pm_reset()
#        return self._pm
#    @pm.setter
#    def pm(self,val):
#        if not val in self.legal_pm:
#            raise ValueError('Unexpected pm arg. Expected one of:',
#                             self.legal_pm,'but got instead',val)
#        self._pm=val
#    def pm_reset(self):
#        self._pm='lin'
#
#    @property
#    def kernel(self):
#        return pm_cmpt(self._kernel,self.pm)
#        #return self._kernel
#    @kernel.setter
#    def kernel(self,ten):
#        if not self.pm == 'lin':
#            print('this shouldn\'t happen')
#        self._kernel=ten
#
#    def __call__(self, inputs, pm='lin'):
#        #Unusually here, you actually need to wrap __call__() instead of call()
#        #"do what you would have before but with
#        #  the specified version of self.kernel and self. bias
#        print('__swd call')
#
#        self.pm=pm# affects self.kernel getter method
#        return super(SignWrapDense,self).__call__(inputs)
#        #return super().__call__(inputs)
#        #return super(self).__call__(inputs)
#        self.pm_reset()
#
#
#
#
#
###First try. write the layer from scratch##
#class SignLinear(layers.Layer):
#    ##Actually holds the weights
#    ##Applies sign components of them as needed
#    #(This is as opposed to having a layer class for each sign component)
#
#    def __init__(self, units=32):
#        super(SignLinear, self).__init__()
#        self.units = units
#
#    def build(self, input_shape):
#        self.w = self.add_weight(shape=(input_shape[-1],self.units),
#                                initializer='random_normal',
#                                trainable=True)
#        self.b = self.add_weight(shape=(self.units,),
#                                #initializer='random_normal',
#                                initializer='zeros',
#                                trainable=True)
#
#    def call(self, inputs,pm='lin'):
#        '''
#        pm  "plus/minus" says which sign components of the weights to use
#        '''
#        legal_pm=['lin','pos','neg','abs']
#
#        pm_w,pm_b=pm_splits([self.w,self.b],pm)
#
#        return tf.matmul(inputs, pm_w) + pm_b
#
##"Trop"classes take a pair of inputs [f,g]
##equivalent to layer([x,0])
#
###Second try. copy paste code with small change
#from tensorflow.python.ops import standard_ops
#from tensorflow.python.ops import sparse_ops
#from tensorflow.python.eager import context
#from tensorflow.python.ops import gen_math_ops
#from tensorflow.python.ops import math_ops
#from tensorflow.python.ops import nn
#class CustomSignDense(layers.Dense):
#
#    def call( self, inputs, pm='lin'):
#        #get combo plusminus of components
#        kernel,bias=pm_splits([self.kernel,self.bias],pm)
#
#        ###Rest is copypaste from tf.keras.layers docs
#        #with kernel,bias in place of self.kernel,self.bias
#        #https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/core.py#L923-L1087
#        rank = len(inputs.shape)
#        if rank > 2:
#            # Broadcasting is required for the inputs.
#            outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
#            # Reshape the output back to the original ndim of the input.
#            if not context.executing_eagerly():
#                shape = inputs.shape.as_list()
#                output_shape = shape[:-1] + [self.units]
#                outputs.set_shape(output_shape)
#        else:
#            inputs = math_ops.cast(inputs, self._compute_dtype)
#            if K.is_sparse(inputs):
#                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, kernel)
#            else:
#                outputs = gen_math_ops.mat_mul(inputs, kernel)
#        if self.use_bias:
#            outputs = nn.bias_add(outputs, bias)
#        if self.activation is not None:
#            return self.activation(outputs)  # pylint: disable=not-callable
#        return outputs
#
#
##assert isinstance(inputs,list)#pass [x,0] if you want normal behavior
##f,g=inputs
#
#
###  maybe should be  TropFC  to imply it does relu stuff  ##
##class TropLinear(layers.Layer):
##    def __init__(self, units=32):
##        super(TropLinear, self).__init__()
##        self.units = units
##        self.linear = SignLinear(units)
##
##    def build(self, input_shape):
##        self.linear.build(input_shape)
##
##
##    def call(self,inputs,sig='relu',pm='lin'):
##        '''
##        pm  "plus/minus" says which sign components of the weights to use
##                legal_pm=['lin','pos','neg','abs']
##        sig is the binary mask to apply after. defaults to relu binary vector
##
##        outputs*sig should broadcast
##        '''
##        h = self.linear.call(inputs,pm=pm)
##        if sig == 'relu':
##            act = relu(h)
##        else: #binary mask
##            act = h*sig
##
##        return act
#
#
##########################################################################
