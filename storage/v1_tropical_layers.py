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
from utils import relu,batch_inner,batch_outer
from config import get_config

import standard_data
import standard_model

#from standard_data import *
from standard_data import peak,inspect_dataset,tuple_splits
from standard_model import logit_bxe



##For SignDense and SignConv2D, I want them to do exactly the same thing,
#but on call, be able to specify which combo of kernel+ or kernel- to use



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


if __name__=='__main__':
    print('tropical_layers.py')

    x=np.random.rand(4,2).astype(np.float32)
    e=tf.constant([ [1.,0.],[0.,1.] ])
    zx=tf.zeros_like(x)
    ze=tf.zeros_like(e)


    sgnden=SignDense(8,activation=relu)#multi inherit
    sg_lin=sgnden( x , pm='lin')
    sg_pos=sgnden( x , pm='pos')
    sg_neg=sgnden( x , pm='neg')
    print('sg works?:',DIFF(sg_lin, sg_pos-sg_neg))


    tdense=TropicalDense(6)
    t_lin=tdense( [x , zx ] )
    t_pos=tdense( [x , zx ] , 'pos')
    t_neg=tdense( [x , zx ] , 'neg')




