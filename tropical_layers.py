from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
#from keras import backend as K
#from keras.layers import Layer
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

#from utils import prepare_dirs_and_logger,save_config,make_folders
from utils import SH,DIFF,EQUAL #db handy
from utils import relu,batch_inner,batch_outer, fg_diff
from utils import f_minus_g
from utils import outer_inner,cwise_inner

import standard_data
from standard_data import peak,inspect_dataset,tuple_splits
from standard_data import mnist as load_mnist



##For SignDense and SignConv2D, I want them to do exactly the same thing,
#but on call, be able to specify which combo of kernel+ or kernel- to use


'''
presently, cant really modify these and expect to load models built with
previous versions
'''

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

    #def equiv_layer(self,inputs):
    #    #Simply the linear layer without worry about sign components
    #    self.pm_reset()
    #    self.__call__(self,inputs)



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

#class SignConv1D(layers.Dense,SignWrap):
class SignConv1D(SignWrap,layers.Conv1D):
    def __init__(self,*args,**kwargs):
        ##Insist no nonlinearity. Otherwise, pm methods make no sense
        kwargs['activation']=None
        super(SignConv1D,self).__init__(*args,**kwargs)


#SignWrap,Conv2D,and Dense all inherit from Layer
#  but no conflict in method overrides
#multi inherit order doesnt matter because Dense,Conv2D dont override Layer.__call__
#both use "self.kernel" and "self.bias" to access weights
#these get modified by properties/methods kernel(),bias() of SignWrap



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


class TropicalDense(SignDense):
    def __init__(self,*args,**kwargs):
        #super(TropicalDense,self).__init__()
        super(SignDense,self).__init__(*args,**kwargs)
    def fg_layer(self,fg_pair):
        #print('TropDense.call')
        f,g=fg_pair
        f2=self(f,pm='pos') + self(g,pm='neg')
        g2=self(g,pm='pos') + self(f,pm='neg')
        if self.use_bias:
            f2+=self.bias
        return [f2,g2]

class TropicalConv1D(SignConv1D):
    def __init__(self,*args,**kwargs):
        super(TropicalConv1D,self).__init__(*args,**kwargs)
    def fg_layer(self,fg_pair):
        #print('TropConv1D.call')
        f,g=fg_pair
        f2=self(f,pm='pos') + self(g,pm='neg')
        g2=self(g,pm='pos') + self(f,pm='neg')
        if self.use_bias:
            f2+=self.bias
        return [f2,g2]
    #def get_config(self):
    #    return self.equiv_layer.get_config()

class TropicalConv2D(SignConv2D):
    def __init__(self,*args,**kwargs):
        super(TropicalConv2D,self).__init__(*args,**kwargs)
    def fg_layer(self,fg_pair):
        #print('TropConv2D.call')
        f,g=fg_pair
        f2=self(f,pm='pos') + self(g,pm='neg')
        g2=self(g,pm='pos') + self(f,pm='neg')
        if self.use_bias:
            f2+=self.bias
        return [f2,g2]
    #def get_config(self):
    #    return self.equiv_layer.get_config()


class TropicalReLU(layers.ReLU):
    def __init__(self,*args,**kwargs):
        super(TropicalReLU,self).__init__(*args,**kwargs)
    def fg_layer(self,fg_pair):
        #max(f,g)-g = relu(f-g)
        f,g=fg_pair
        return [tf.maximum(f,g),g]


class TropicalFlatten(layers.Flatten):
    def __init__(self,*args,**kwargs):
       super(TropicalFlatten,self).__init__(*args,**kwargs)
    def fg_layer(self,fg_pair):
        f,g=fg_pair
        return [self(f),self(g)]#flatten both




class TropicalMaxPool1D(layers.MaxPooling1D):
    def __init__(self,*args,**kwargs):
        kwargs['padding']='valid' #need constant patch size to sum_pool
        super(TropicalMaxPool1D,self).__init__(*args,**kwargs)
        self.ave_pool=layers.AveragePooling1D(*args,**kwargs)
        #self.max_pool=layers.MaxPooling1D(*args,**kwargs)#same as MaxPool1D
        self.n_pool  =np.prod(self.ave_pool.pool_size)
        #self.equiv_layer=self.max_pool
    def fg_layer(self,fg_pair):
        f,g=fg_pair
        #sum pool
        G=self.n_pool * self.ave_pool( g )
        f2 = G + self( f-g )#G+maxpool(f-g)
        g2 = G
        return [f2,g2]


class TropicalMaxPool2D(layers.MaxPooling2D):
    def __init__(self,*args,**kwargs):
        kwargs['padding']='valid' #need constant patch size to sum_pool
        super(TropicalMaxPool2D,self).__init__(*args,**kwargs)
        self.ave_pool=layers.AveragePooling2D(*args,**kwargs)
        #self.max_pool=layers.MaxPooling2D(*args,**kwargs)#same as MaxPool2D
        self.n_pool  =np.prod(self.ave_pool.pool_size)
        #self.equiv_layer=self.max_pool
    def fg_layer(self,fg_pair):
        f,g=fg_pair
        #sum pool
        G=self.n_pool * self.ave_pool( g )
        f2 = G + self( f-g )#G+maxpool(f-g)
        g2 = G
        return [f2,g2]


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
        #self.equiv_layer=self.identity
    def call(self,inputs):
        inputs=self.identity(inputs)
        return [inputs,tf.zeros_like(inputs)]


class TropicalModel(keras.Model):
#class TropicalModel(object):
    '''
    A wrapper that exposes/composes methods in Tropical layer variants
    '''
    def __init__(self,*args,**kwargs):
        super(TropicalModel,self).__init__(*args,**kwargs)

#    def fg_call_nograd(self,inputs):
#        x=inputs
#        fg_x = [x,tf.zeros_like(x)]   # or not?
#        for L in self.layers:
#            fg_x=L.fg_layer(fg_x)
#        return fg_x

#f_minus_g  #use me

    def fg_call(self,inputs):#,return_fg=False
        ##use tf layers to eval tropical polynomials f(x),g(x)#fg_x
        ##perhaps also, return local linear version of f,g
        x=inputs

        #grad tape needs tensor, wont work with numpy
        if isinstance(x,np.ndarray):
            x=tf.constant(x)

        #a handy hack
        #Exploit tf automatic differentiation to find local affine representation
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            ##---  outputs=func(x)  ---##
            #fg=TropicalEmbed()(x)  #to use layer like this
            fg_x = [x,tf.zeros_like(x)]   # or not?
            for L in self.layers:
                fg_x=L.fg_layer(fg_x)
            ##---  outputs=func(x)  ---##

        #not sure if expensive
        fg_Linear=[tape.gradient(out,x) for out in fg_x ] #each of shape x
        del tape

        fg_Linear_x=[batch_inner(x,lin) for lin in fg_Linear]
        fg_Bias=[y-linX for y,linX in zip(fg_x,fg_Linear_x)]


        ##Three length=2 lists
        return fg_x,fg_Linear,fg_Bias

    #added 11-22
    def trf(self,inputs):#"Tropical Rational Function"
        fg_x=self.fg_call(inputs)[0]
        return f_minus_g( fg_x )

    def validate_tropical(self,inputs):
        '''verify tropical calls are equivalent'''
        net_x=self.__call__(inputs)
        fg_x,fg_Linear,fg_Bias=self.fg_call(inputs)

        #fg_diff=lambda fg:fg[0]-fg[1]
        trop_x=fg_diff(fg_x)
        print('Numerical error in tropical forwardpass:',DIFF(net_x,trop_x))

        net_linear=fg_diff(fg_Linear)
        net_bias  =fg_diff(fg_Bias)
        #local linear predict
        ll_x = batch_inner(inputs,net_linear)+net_bias

        print('Numerical error in local linear computaiton',DIFF(net_x,ll_x))

        return net_x,trop_x,ll_x

class TropicalSequential(TropicalModel,keras.Sequential):
    '''
    A sequential model that is aware its layers are TropicalLayers
    '''
    def __init__(self,*args,**kwargs):
        super(TropicalSequential,self).__init__(*args,**kwargs)


#class TropicalPolynomial(layers.Layer):
class TropicalRational(layers.Layer):
    def __init__(self,kernel,bias,**kwargs):
        '''
        Layer for computing explicit different of piecewise linear convex fns
        Difference of two convex fns, each a max of f linear functions

        kernel: array of shape  (f,2) + input_shape
        bias  : array of shape  (f,2)

        '''
        #super(TropicalPolynomial,self).__init__(**kwargs)
        super(TropicalRational,self).__init__(**kwargs)
        assert(kernel.shape[1]==2)

        self.kernel=tf.constant(kernel)
        self.bias  =tf.constant(bias  )

        rk=tf.rank(self.kernel)
        #self.dot_ax = rk - tf.range(rk-2) - 1
        self.dot_ax  = -1* tf.range(rk-2) - 1

        #self.F=F  implement this way if |F| != |G|
        #self.G=G

    def build(self,input_shape):
        super(TropicalRational, self).build(input_shape)

    def call(self,inputs):
        '''
        inputs is a shape  (bhsz ,) + input_shape
        shaped array representing a batch of inputs.
        '''
        #for every pair x in Batch, (v,w) in FxG
        # inner product over the trailing matching dims,
        # which represent the input_shape by convention

        #if tf.rank(inputs)+1==tf.rank(self.kernel):
        #    inputs=tf.expand_dims(inputs,1)

        ax=self.dot_ax
        gram=tf.tensordot(inputs,self.kernel,axes=[ax,ax])
        #gram  = outer_inner( inputs, self.kernel ) + self.bias #bhsz x f x 2
        #polys = np.max( gram , axis=1 ) #bhsz x 2
        polys = tf.reduce_max( gram , axis=1 ) #bhsz x 2
        sgnpolys = polys*tf.constant([1.,-1.])
        return tf.reduce_sum(sgnpolys,axis=-1)

        #f,g = tf.split(polys,num_or_size_splits=2,axis=1)
        #return f - g  #bhsz x 1 


        #opred = f_minus_g( polys )        #bhsz x 1
        #return opred

#    def compute_output_shape(self, input_shape ):
#        #assert isinstance(input_shape, list)
#        bhsz=input_shape[0]
#        return tf.TensorShape([bhsz,1])

    def get_config(self):
        #config = super(TropicalPolynomial,self).get_config()
        config = super(TropicalRational,self).get_config()
        config.update({'kernel':self.kernel,
                       'bias'  :self.bias  })
        return config
    ##def from_config(cls

#https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#compute_output_shape

#https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/engine/training.py#L342

tropical_objects={
    'TropicalMaxPool1D' :TropicalMaxPool1D,
    'TropicalMaxPool2D' :TropicalMaxPool2D,
    'TropicalReLU'      :TropicalReLU,
    'TropicalFlatten'   :TropicalFlatten,
#    'TropicalEmbed'     :TropicalEmbed,
    'TropicalConv1D'    :TropicalConv1D,
    'TropicalConv2D'    :TropicalConv2D,
    'TropicalDense'     :TropicalDense,
#    'TropicalModel'     :TropicalModel,
    'TropicalSequential':TropicalSequential,
    'TropicalRational'  :TropicalRational,
}


tropical_equivalents={
    layers.MaxPooling2D:TropicalMaxPool2D,
    layers.ReLU        :TropicalReLU,
    layers.Flatten     :TropicalFlatten,
    #TropicalEmbed,
    layers.Conv2D      :TropicalConv2D,
    layers.Dense       :TropicalDense,
}

##infer trop layers from numpy arrays 
#def def lyr_from_wts(weights)
#I had some old fully-connected models I wanted to test out
def wts2lyr(weights,*args,**kwargs):
    kernel,bias=weights
    #Kinit=keras.initializers.Constant(kernel)
    #Binit=keras.initializers

    filters=kernel.shape[-1]#units
    ndim=kernel.ndim
    if ndim == 2:#dense
        input_shape=(kernel.shape[0],)
        layer=TropicalDense(units=filters,input_shape=input_shape,
                            *args,**kwargs)#,weights=weights)
        layer.build(input_shape)
        layer.set_weights(weights)

    elif ndim == 4:
        raise ValueError('kernel.ndim=4 but Conv2D inference not implemented')
    else:
        raise ValueError('Expected weights[0].ndim in [2,4] but got',
                         kernel.ndim,'with shape',kernel.shape,
                         'Unclear how to interpret as FC or Conv')

    return layer



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
    t_lin=tdense.fg_layer( [x , zx ] )


    datasets,info=load_mnist()
    #datasets,info=load_fashion_mnist()
    #datasets,info=load_cifar10()
    train_data,test_data=tuple_splits(datasets)
    info_shape=info.features['image'].shape
    iis=info.input_shape
    input_shape=tf.TensorShape(iis)
    db_bat=peak(datasets['train'].batch(20))
    X=db_bat['x']
    Y=db_bat['y']# shape (bsz,)
    O=tf.zeros_like(X)#looks like 0. is an Oh.
    l=tf.ones_like(X)#kinda looks like a 1
    db_bat25=peak(datasets['train'].batch(25))
    X25=db_bat25['x']
    X20=X
    ds_train=datasets['train']
    ds_input=ds_train.map(lambda e:e['x']).batch(1)
    input_shape=db_bat['x'].shape[1:]#equiv to tf.TS(info.is)

    ###prep training###
    train_size=info.splits['train'].num_examples
    test_size=info.splits['test'].num_examples
    buf=np.int(train_size*0.1)
    ####end data####



    ksz=(3,3)#kernel size
    sds=(2,2)#strides
    model=keras.Sequential([
        keras.layers.Conv2D(32,ksz,sds,activation='relu',name='L1',
                                 input_shape=input_shape),
        keras.layers.Conv2D(64,ksz,sds,activation='relu',name='L2'),
        keras.layers.Conv2D(64,ksz,sds,activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(1),
    ])

    tmodel=keras.Sequential([
        TropicalConv2D(32,ksz,sds,name='L1',input_shape=input_shape),
        TropicalReLU(),
        TropicalConv2D(64,ksz,sds,name='L2'),#,activation='relu',name='L2'),
        TropicalReLU(),
        TropicalConv2D(64,ksz,sds),#,activation='relu'),
        TropicalReLU(),
        TropicalFlatten(),
        TropicalDense(64),#,activation='relu'),
        TropicalReLU(),
        TropicalDense(1),
    ])

    unbuilt_tmodel=keras.Sequential([
        TropicalConv2D(32,ksz,sds,name='L1'),
        TropicalReLU(),
        TropicalConv2D(64,ksz,sds,name='L2'),#,activation='relu',name='L2'),
        TropicalReLU(),
        TropicalConv2D(64,ksz,sds),#,activation='relu'),
        TropicalReLU(),
        TropicalFlatten(),
        TropicalDense(64),#,activation='relu'),
        TropicalReLU(),
        TropicalDense(1),
    ])


#The added layer must be an instance of class Layer 

