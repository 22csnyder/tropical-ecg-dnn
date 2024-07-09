from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import oldstyle
from oldstyle import toydata as v1data

import os
import pandas as pd
import numpy as np

import scipy.io as io
import biosppy

'''
Standardization of datasets pipeline

Standardized Params:
'idx': key for id asso to each input
'x'  : key for input
'y'  : key for binary label (ndim=1)
'train'
'test'
info.input_shape : pass to model input_shape of element['x']
'''

#Handy
#dataset.element_spec

##x,y,idx are protected words
def add_index(dataset):
    def flatten_index(index,elem):
        assert('index' not in elem.keys() )
        elem['idx']=index
        return elem
    enum_data=dataset.enumerate()
    return enum_data.map(flatten_index)

#def zip_dict(tup_datasets):
#    #


def tuple_splits(datasets):
    '''
    Uses 'train' and 'test' keys of datasets
    For each, return a new dataset of ordered pairs ('x','y')

    formats train test the way that keras.model.fit would like
    requires two special keys: 'x', 'y'
    '''
    #Makes format read for training
    target_data={k:v.map(lambda e:(e['x'],e['y']) ) for k,v in datasets.items()}
    train_data=target_data['train']
    test_data =target_data['test']
    return train_data,test_data

def float_image(elem):
    elem['x']=tf.cast(elem['image'],tf.float32)/255.
    return elem

##---MNIST---##
def mnist():
    def binarize(elem):
        elem['y']=tf.cast( tf.math.greater(elem['label'],4), tf.uint8)
        return elem
    #WARN may shuffle train data#
    builder=tfds.builder('mnist')
    info=builder.info
    info.input_shape=info.features['image'].shape
    datasets=builder.as_dataset(shuffle_files=False)
    #datasets,info=tfds.load('mnist',with_info=True)
    datasets={k:v.apply(add_index) for k,v in datasets.items()}
    datasets={k:v.map(float_image) for k,v in datasets.items()}
    datasets={k:v.map(binarize)    for k,v in datasets.items()}
    return datasets,info


def mnist_511split():
    #same as mnist but includes a validation split
    builder=tfds.builder('mnist')
    info=builder.info
    info.input_shape=info.features['image'].shape
    train50k,valid10k = tfds.Split.TRAIN.subsplit([5,1])
    test10k=tfds.Split.TEST

    ds_train=builder.as_dataset(shuffle_files=False,split=train50k)#49800#measurement
    ds_valid=builder.as_dataset(shuffle_files=False,split=valid10k)#10200
    ds_test =builder.as_dataset(shuffle_files=False,split=test10k )#10000
    datasets={'train':ds_train,'valid':ds_valid,'test':ds_test}

    info.train_size=49800#This is new
    info.valid_size=10200#needed when you do splits
    info.test_size =10000

    datasets={k:v.apply(add_index) for k,v in datasets.items()}
    datasets={k:v.map(float_image) for k,v in datasets.items()}
    datasets={k:v.map(binarize)    for k,v in datasets.items()}
    return datasets,info



def fashion_mnist():
    #10 classes in fashion mnist
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle, boot']

    def binarize(elem):
        elem['y']=tf.cast( tf.math.greater(elem['label'],4), tf.uint8)
        return elem

    builder=tfds.builder('fashion_mnist')
    info=builder.info
    info.input_shape=info.features['image'].shape

    datasets=builder.as_dataset(shuffle_files=False)
    datasets={k:v.apply(add_index) for k,v in datasets.items()}
    datasets={k:v.map(float_image) for k,v in datasets.items()}
    datasets={k:v.map(binarize)    for k,v in datasets.items()}

    return datasets,info

###---Celeb_A---###
def proc_celeb_a(elem):
    x=tf.cast(elem['image'],tf.float32)/255.
    #Central crop [218x178]->[108,108]
    x=tf.image.resize_with_crop_or_pad(x, 108,108)
    x=tf.image.resize(x,[64,64], method='area')
    elem['x']=x
    #elem['y']=tf.cast( elem['attributes']['Attractive'] , tf.uint8 )
    elem['y']=tf.cast( elem['attributes']['Black_Hair'], tf.uint8)
    #elem['y']=tf.cast( elem['attributes']['Wearing_Lipstick'], tf.uint8)
    return elem

def celeb_a():
    #WARN may shuffle train data#
    builder=tfds.builder('celeb_a')
    info=builder.info
    ##info.input_shape=info.features['image'].shape# initial
    datasets=builder.as_dataset(shuffle_files=False)
    #datasets,info=tfds.load('celeb_a',with_info=True)
    datasets=tf.nest.map_structure(add_index                   ,datasets)

    datasets=tf.nest.map_structure(lambda D:D.map(proc_celeb_a),datasets)#rs
    info.input_shape=(64,64)+info.features['image'].shape[2:]

    return datasets,info


###---CIFAR10---###
def cifar10():
    def binarize(elem):
        elem['y']=tf.cast( tf.math.greater(elem['label'],4), tf.uint8)
        return elem
    #datasets,info=tfds.load('cifar10',with_info=True)#worried about shuffle on train
    builder=tfds.builder('cifar10') #alt method instead
    info=builder.info
    info.input_shape = info.features['image'].shape
    datasets=builder.as_dataset(shuffle_files=False)

    datasets={k:v.apply(add_index) for k,v in datasets.items()}
    datasets={k:v.map(float_image) for k,v in datasets.items()}
    datasets={k:v.map(binarize)    for k,v in datasets.items()}
    return datasets,info

def frogs_vs_ships():
    ##Frogs vs Ships##
    datasets,info=cifar10()
    lab0,lab1=map(info.features['label'].encode_example,['frog','ship'])
    def cifar_filter(elem):
        return tf.logical_or( tf.equal(elem['label'],lab0),
                       tf.equal(elem['label'],lab1) )
    def label2y(elem):
        elem['y']=tf.cast(
            tf.math.equal(elem['label'],lab0),
            tf.uint8)
        return elem
    datasets={k:v.filter(cifar_filter).map(label2y) for k,v in datasets.items()}
    return datasets,info


def wrap_oldstyle_data( data_fn ):
    def wrapped_data_fn(**kwargs):
        kwargs['return_numpy']=True
        npdict=data_fn(**kwargs)
        npX=npdict['input'].astype(np.float32)
        npY=(0.5+0.5*npdict['label']).astype(np.uint8).flatten()
        ds=tf.data.Dataset.from_tensor_slices( {'x':npX, 'y':npY})
        ds=ds.apply(add_index)
        return ds
    return wrapped_data_fn

##data1 through data3 are from previous paper
def v1_data1():#linearly seperable
    #Pub_Model_0504_233305_D1A3
    v1_uniforce=wrap_oldstyle_data(v1data.uniforce)
    datasets={
        'train':v1_uniforce(),
        'test' :v1_uniforce(seed=25),
        'valid':v1_uniforce(seed=26)}
    def info():
        pass#hack
    info.train_size=40#This is new
    info.valid_size=40#needed when you do splits
    info.test_size =40
    return datasets,info


def v1_noisy1_data1():
    #Fig_Model_0505_005701_noise1D1A3#*
    v1_noisy1_uniforce=wrap_oldstyle_data(v1data.noisy1_uniforce)
    v1_uniforce=wrap_oldstyle_data(v1data.uniforce)

    datasets={
        'train':v1_noisy1_uniforce(),
        'test' :v1_uniforce(seed=25),
        'valid':v1_uniforce(seed=26)}
    def info():
        pass#hack
    info.train_size=48
    info.valid_size=40
    info.test_size =40
    info.input_shape=(2,)

    return datasets,info

def v1_noisy2_data1():
    #Fig_Model_0505_003620_noise2D1A3#*
    v1_noisy2_uniforce=wrap_oldstyle_data(v1data.noisy2_uniforce)
    v1_uniforce=wrap_oldstyle_data(v1data.uniforce)

    datasets={
        'train':v1_noisy2_uniforce(),
        'test' :v1_uniforce(seed=25),
        'valid':v1_uniforce(seed=26)}
    def info():
        pass#hack
    info.train_size=45
    info.valid_size=40
    info.test_size =40
    info.input_shape=(2,)

    return datasets,info

def v1_noisy3_data1():
    #Fig_Model_0505_012310_noise3D1A3
    v1_noisy3_uniforce=wrap_oldstyle_data(v1data.noisy3_uniforce)
    v1_uniforce=wrap_oldstyle_data(v1data.uniforce)

    datasets={
        'train':v1_noisy3_uniforce(),
        'test' :v1_uniforce(seed=25),
        'valid':v1_uniforce(seed=26)}
    def info():
        pass#hack
    info.train_size=117
    info.valid_size=40
    info.test_size =40
    info.input_shape=(2,)

    return datasets,info






####These were for encorporating the patient data into physionet###
#def load_data_physionet2017():
#    'do train test split, load. everythign except dataset'
#
#    template_run='templates1'
#    data_dir='./data/'
#    raw_data_dir =data_dir+'physionet2017/training2017/'
#    proc_data_dir=data_dir+'processed_physionet2017/'
#    csv_pth=os.path.join(raw_data_dir,'REFERENCE.csv')
#    templates_dir=proc_data_dir+template_run
#
#    #load patient dataset
#    DF=pd.read_csv(csv_pth,header=None,names=['key','label'])
#    DF['raw data file']=DF['key'].map(lambda k:os.path.join(raw_data_dir,k+'.mat'))
#    DF['templates file']=DF['key'].map(lambda k:os.path.join(templates_dir,k+'_templates.csv'))
#
#    #all_templates= np.load('./tmp/allTemplates.npy')#
#    all_templates= np.load('./tmp/allTemplates.npy',allow_pickle=True)
#    assert len(all_templates)==len(DF)
#
#    N=len(all_templates)
#    p_train=0.8
#    n_train=np.int(p_train*N)
#    n_test =N-n_train
#
#    np.random.seed(22)#same split every time
#    perm=np.random.permutation(N)
#    train_idx=perm[:n_train]
#    test_idx=perm[n_train:]
#
#    train_data=(DF.loc[train_idx],all_templates[train_idx])
#    test_data =(DF.loc[test_idx], all_templates[test_idx])
#
#    return train_data,test_data
#
#def templates_to_dataset( ecg_data ):
#    DF,all_templates = ecg_data
#    assert len(all_templates)==len(DF)
#
#    pat_records=DF.to_dict('list')#{'key':['A001','A002'..],'label':['N','A'..]}
#    ds_Patients=tf.data.Dataset.from_tensor_slices(pat_records)
#    ds_template_stacks=tf.data.Dataset.from_tensor_slices({'template':all_templates})
#
#    #combine the two
#    ds_zipped=tf.data.Dataset.zip((ds_Patients,ds_template_stacks))
#    def flatten_zip(e1,e2):
#        e3={}
#        e3.update(e1)
#        e3.update(e2)
#        return e3
#    def ungroup_templates(eP,eT):
#        T=tf.data.Dataset.from_tensor_slices(eT)#single row ds. pat info
#        T=T.apply(add_index)#maybe shuffle here
#        P=tf.data.Dataset.from_tensors(eP).repeat()#each row a beat of pat
#        #P=tf.data.Dataset.from_tensor_slices(eP).repeat()#each row a beat of pat
#        Z=tf.data.Dataset.zip((P,T))
#        #Z=tf.data.Dataset.zip((P,P))
#        #Z=tf.data.Dataset.zip((T,T))
#        return Z.map(flatten_zip)
#    #will pick one template at a time from each patient dataset
#    ds_data=ds_zipped.interleave( ungroup_templates )
#
#    return ds_data
#
#def physionet2017():
#    train_data,test_data = load_data_physionet2017()
#    train_ds  ,test_ds   = map(templates_to_dataset,[train_data,test_data])
#    datasets={
#        'train':train_ds,
#        'test' :test_ds}
#
#    #def not_noisy(elem):
#    #    return tf.not_equal(elem['label'],'~')
#    def normal_v_other(elem):
#        is_normal=tf.equal(elem['label'],'N')
#        elem['y']=tf.cast(is_normal,tf.uint8)
#        elem['x']=tf.cast(elem['template'],tf.float32)
#        elem['x']=tf.reshape(elem['x'],[-1,1])#(150,1) not (150,)
#        return elem
#    #ds_data=ds_data.filter(not_noisy).map(normal_v_other)
#
#    datasets={k:v.map(normal_v_other) for k,v in datasets.items()}
#
#    def info():
#        pass#hack
#    #info.train_size=264368
#    #tf.data.experimental.cardinality(test_ds)
#    info.train_size=211482
#    #info.valid_size=500
#    info.test_size =52886
#    info.input_shape=(150,1)
#
#    return datasets,info
####These were for encorporating the patient data into physionet###



#Not noisy data flow
#WARN; this splits on templates, not patients
def ecg_templates():
    data_dir='./data/'
    proc_data_dir=data_dir+'processed_physionet2017/'
    trainX=np.load( proc_data_dir+ 'trainX.npy')
    trainY=np.load( proc_data_dir+ 'trainY.npy')
    testX =np.load( proc_data_dir+ 'testX.npy')
    testY =np.load( proc_data_dir+ 'testY.npy')

    train_ds = tf.data.Dataset.from_tensor_slices( {'template':trainX,'label':trainY} )
    test_ds = tf.data.Dataset.from_tensor_slices( {'template':testX,'label':testY} )
    datasets={ 'train':train_ds, 'test' :test_ds}
    datasets={k:v.apply(add_index) for k,v in datasets.items()}

    def normal_v_other(elem):
        is_normal=tf.equal(elem['label'],'N')
        elem['y']=tf.cast(is_normal,tf.uint8)
        elem['x']=tf.cast(elem['template'],tf.float32)
        elem['x']=tf.reshape(elem['x'],[-1,1])#(150,1) not (150,)
        return elem
    #ds_data=ds_data.filter(not_noisy).map(normal_v_other)
    datasets={k:v.map(normal_v_other) for k,v in datasets.items()}
    def info():
        pass#hack
    info.train_size=276216
    info.test_size =69055
    info.input_shape=(150,1)
    return datasets,info
def not_noisy(elem):
    return tf.not_equal(elem['label'],'~')
def ecg_notnoisy():#0.55% normal #57%N, 32%O, 10%A #in train
    datasets,info=ecg_templates()
    datasets={k:ds.filter(not_noisy) for k,ds in datasets.items()}
    info.train_size=267831
    info.test_size=66948
    return datasets,info

def not_af(elem):
    return tf.not_equal(elem['label'],'A')
def ecg_notnoisy_noraf():
    datasets,info = ecg_notnoisy()
    datasets={k:ds.filter(not_af) for k,ds in datasets.items()}
    return datasets, info


###debugging and interactive utilities###
def inspect_dataset(name):
    '''
    Just for debug purposes. Quickly see what a batch is like
    '''
    train_data=tfds.load(name,split='train')
    batched=train_data.batch(32).take(1)
    batch=next(iter(batched))
    print('Keys:',batch.keys())
    return batch

def peak(dataset):
    '''also mainly for debug. see what it looks like'''
    return next(iter(dataset))


if __name__=='__main__':#for db

    #datasets,info=physionet2017()
    datasets,info=ecg_templates()
    train_ds=datasets['train']
    test_ds =datasets['test']
    bat=peak(train_ds.batch(20))
    X20=bat['x']




#    #The idea will be to pass the raw data dir containing orig files
#    #Then depending on the preproc method, pass the corresp templates dir
#    template_run='templates1'
#
#    data_dir='./data/'
#    raw_data_dir =data_dir+'physionet2017/training2017/'
#    proc_data_dir=data_dir+'processed_physionet2017/'
#    csv_pth=os.path.join(raw_data_dir,'REFERENCE.csv')
#    templates_dir=proc_data_dir+template_run
#
#    DF=pd.read_csv(csv_pth,header=None,names=['key','label'])
#    DF['raw data file']=DF['key'].map(lambda k:os.path.join(raw_data_dir,k+'.mat'))
#
#    #!@
#    DF['templates file']=DF['key'].map(lambda k:os.path.join(templates_dir,k+'_templates.csv'))
#
#    #DB#pat_records=DF.head(10).to_dict('list')#{'key':['A001','A002'..],'label':['N','A'..]}
#    pat_records=DF.to_dict('list')#{'key':['A001','A002'..],'label':['N','A'..]}
#
#    #pat_ecgs=[io.loadmat(fn)['val'][0] for fn in DF['raw data file']]#not all same len
#    #pat_records['ecg']=pat_ecgs
#    #I guess dont need to load these if templates are preprocessed and saved
#
#    #ds_Patients=tf.data.Dataset.from_tensor_slices(pat_records[:15])
#    ds_Patients=tf.data.Dataset.from_tensor_slices(pat_records)
#    #ds_Patients=ds_Patients.apply(add_index)
#    #pat_templates = [np.loadtxt(fn) for fn in pat_records['templates file']]#slow
#    all_templates= np.load('./tmp/allTemplates.npy')#
#    assert len(all_templates)==len(DF)
#    ds_template_stacks=tf.data.Dataset.from_tensor_slices({'template':all_templates})
#
#    ds_zipped=tf.data.Dataset.zip((ds_Patients,ds_template_stacks))
#    #ds_combined=ds_combined.map(lambda e1,e2:e1.update(e2))
#
#    def flatten_zip(e1,e2):
#        e3={}
#        e3.update(e1)
#        e3.update(e2)
#        return e3
#
#
#    def ungroup_templates(eP,eT):
#        T=tf.data.Dataset.from_tensor_slices(eT)#single row ds. pat info
#        T=T.apply(add_index)#maybe shuffle here
#        P=tf.data.Dataset.from_tensors(eP).repeat()#each row a beat of pat
#        #P=tf.data.Dataset.from_tensor_slices(eP).repeat()#each row a beat of pat
#        Z=tf.data.Dataset.zip((P,T))
#        #Z=tf.data.Dataset.zip((P,P))
#        #Z=tf.data.Dataset.zip((T,T))
#        return Z.map(flatten_zip)
#
#    #will pick one template at a time from each patient dataset
#    ds_data=ds_zipped.interleave( ungroup_templates )


#def physionet2017():
#    #could use tf.io.decode_csv to load from text files
#    template_run='templates1'
#
#    data_dir='./data/'
#    raw_data_dir =data_dir+'physionet2017/training2017/'
#    proc_data_dir=data_dir+'processed_physionet2017/'
#    csv_pth=os.path.join(raw_data_dir,'REFERENCE.csv')
#    templates_dir=proc_data_dir+template_run
#
#    #load patient dataset
#    DF=pd.read_csv(csv_pth,header=None,names=['key','label'])
#    DF['raw data file']=DF['key'].map(lambda k:os.path.join(raw_data_dir,k+'.mat'))
#    DF['templates file']=DF['key'].map(lambda k:os.path.join(templates_dir,k+'_templates.csv'))
#    pat_records=DF.to_dict('list')#{'key':['A001','A002'..],'label':['N','A'..]}
#    ds_Patients=tf.data.Dataset.from_tensor_slices(pat_records)
#
#    #pat_ecgs=[io.loadmat(fn)['val'][0] for fn in DF['raw data file']]#not all same len
#    #pat_records['ecg']=pat_ecgs
#
#    #load preproc templates dataset
#    #ds_Patients=tf.data.Dataset.from_tensor_slices(pat_records[:15])
#    #ds_Patients=ds_Patients.apply(add_index)
#    #pat_templates = [np.loadtxt(fn) for fn in pat_records['templates file']]#slow
#    all_templates= np.load('./tmp/allTemplates.npy')#
#    assert len(all_templates)==len(DF)
#    ds_template_stacks=tf.data.Dataset.from_tensor_slices({'template':all_templates})
#
#
#    #combine the two
#    ds_zipped=tf.data.Dataset.zip((ds_Patients,ds_template_stacks))
#    def flatten_zip(e1,e2):
#        e3={}
#        e3.update(e1)
#        e3.update(e2)
#        return e3
#    def ungroup_templates(eP,eT):
#        T=tf.data.Dataset.from_tensor_slices(eT)#single row ds. pat info
#        T=T.apply(add_index)#maybe shuffle here
#        P=tf.data.Dataset.from_tensors(eP).repeat()#each row a beat of pat
#        #P=tf.data.Dataset.from_tensor_slices(eP).repeat()#each row a beat of pat
#        Z=tf.data.Dataset.zip((P,T))
#        #Z=tf.data.Dataset.zip((P,P))
#        #Z=tf.data.Dataset.zip((T,T))
#        return Z.map(flatten_zip)
#    #will pick one template at a time from each patient dataset
#    ds_data=ds_zipped.interleave( ungroup_templates )
#
#
#    def not_noisy(elem):
#        return tf.not_equal(elem['label'],'~')
#    def normal_v_other(elem):
#        is_normal=tf.equal(elem['label'],'N')
#        elem['y']=tf.cast(is_normal,tf.uint8)
#        elem['x']=tf.cast(elem['template'],tf.float32)
#        elem['x']=tf.reshape(elem['x'],[-1,1])#(150,1) not (150,)
#        return elem
#    ds_data=ds_data.filter(not_noisy).map(normal_v_other)
#
#
#    datasets={'train':ds_data,
#              'test' :ds_data.take(500),
#              'valid':ds_data.take(500)}#for now
#    def info():
#        pass#hack
#    info.train_size=264368
#    info.valid_size=500
#    info.test_size =500
#    info.input_shape=(150,1)
#
#    return datasets,info


#def expdim(elem):
#    elem['template']=tf.reshape(elem['template'],[-1,1])
#    return elem
#rs_data=ds_data.map(expdim)



#    def ungroup_templates(e):
#        T=tf.data.Dataset.from_tensor_slices(#single row ds. pat info
#            {'beat_idx':e.pop('idx'),
#             'template':e.pop('template')})
#        P=tf.data.Dataset.from_tensors(e).repeat()#each row a beat of pat
#        return tf.data.Dataset.zip((P,T))



    #works
#    DS=tf.data.Dataset
#    ds1=DS.from_tensor_slices({'A':range(5)}).repeat()
#    ds2=DS.from_tensor_slices({'B':[-1]*7}).repeat()
#    z12=DS.zip((ds1,ds2))
#    fz=z12.map(flatten_zip)








#--

#    #key2fname = lambda k : os.path.join(data_dir,k+'.mat')
#    #fname2arr = lambda f : io.loadmat( fname )['val'][0]
#    mat2ecg   = lambda m : biosppy.ecg.ecg(m,sampling_rate=200.)
#    def load_raw_ecgs(csv_row):
#        e={}#element
#        e['key'],e['label']=csv_row[0],csv_row[1]
#
#        #fname=key2fname(e['key'].numpy())
#        #arr1d=io.loadmat( e['file'] )['val'][0]
#
#
#        e['raw data file']=key2fname(e['key'].numpy())
#        e['raw ecg']=io.loadmat( e['file'] )['val'][0]
#        #e['beat templates file']=
#        return e
#    import warnings
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")#biosppy does some deprecated things
#        ECG=biosppy.ecg.ecg(f,sampling_rate=200.0,show=False)


#    proc_line=lambda l:tf.strings.split(l,',')#l= 'A00008,O'
#    ds_csv=tf.data.TextLineDataset(csv_pth).map(proc_line)#('A00008','0')
#    ds_raw=ds_csv.map(load_raw_ecgs)
#    def ref2data(csv_line):
#        key,label= csv_line.split(',')
#        mat_fn=key2fname(key)
#        signal=io.loadmat(mat_fn)['val'][0]


#--

    #try loading prev data

    #ds splits covered here;
#https://github.com/tensorflow/datasets/blob/e53f3af5997bd0af9f7e61de3b8c98d8254e07b6/docs/splits.md

    #try splitting mnist#

#    test10k=tfds.Split.TEST
#    builder=tfds.builder('mnist')
#    info=builder.info
#    info.input_shape=info.features['image'].shape
#    train50k,valid10k = tfds.Split.TRAIN.subsplit([5,1])
#
#    ds_train=builder.as_dataset(shuffle_files=False,split=train50k)
#    ds_valid=builder.as_dataset(shuffle_files=False,split=valid10k)
#    ds_test =builder.as_dataset(shuffle_files=False,split=test10k )
#    datasets={'train':ds_train,'valid':ds_valid,'test':ds_test}
#
#    ds50,io50=tfds.load('mnist',shuffle_files=False,split=train50k,with_info=True)
#    ds60,io60=tfds.load('mnist',shuffle_files=False,split=tfds.Split.TRAIN,with_info=True)

#--

#
#    cnt60=0
#    for bh in ds60.batch(200):
#        cnt60+=len(bh['image'])
#
#    cnt50=0
#    for bh in ds50.batch(50):
#        cnt50+=len(bh['image'])  
#
#
#
#    #datasets={'train':builder.as_dataset(shuffle_files=False,split=train40k),
#    #          'valid':builder.as_dataset(shuffle_files=False,split=valid10k),
#    #          'test' :builder.as_dataset(shuffle_files=False,split=test10k ),
#    #         }
#    #datasets=builder.as_dataset(shuffle_files=False)
#
#
#    stophere
#
#
#    #db: getting input_shape
#
#    #datasets,info=celeb_a()
#    #datasets,info=mnist()
#    datasets,info=frogs_vs_ships()
#    #datasets,info=fashion_mnist()
#    #datasets,info=cifar10()
#    train_data,test_data=tuple_splits(datasets)
#
#    info_shape=info.features['image'].shape
#    iis=info.input_shape
#
#    #---debug datasets---#
#    db_bat=peak(datasets['train'].batch(20))
#    X=db_bat['x']
#    db_bat25=peak(datasets['train'].batch(25))
#    X25=db_bat25['x']
#    X20=X
#
#    ds_train=datasets['train']
#    ds_input=ds_train.map(lambda e:e['x']).batch(1)
#    #input_shape=[None]+db_bat['x'].shape[1:]
#    input_shape=db_bat['x'].shape[1:]
#    #input_shape=[None]+tf.compat.v1.data.get_output_shapes(ds_input)
#    #input_shape=[None]+ds_input.output_shapes[1:]
#    #ds_input.output_shapes
#
#
#    #datasets,info=tfds.load('mnist',with_info=True)
#    #datasets=tfds.load('mnist')
#
#

    print('finished!')

