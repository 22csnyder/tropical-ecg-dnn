from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import six
import json
import glob2

import standard_data as data_module
import standard_model as model_module

def str2bool(v):
    return v is True or v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

data_arg=add_argument_group('Data')
misc_arg=add_argument_group('Misc')
model_arg=add_argument_group('Model')
train_arg = add_argument_group('Training')
exp_arg = add_argument_group('Experiment')


#data_arg.add_argument('--dataset',type=str,choices=['mnist','cifar10'],default='mnist', help='''what dataset to run on''')


##--------------------    Data Args    ------------------------##
data_arg.add_argument('--batch_size',type=int,default=64)
data_arg.add_argument('--data',type=str,default='',choices=dir(data_module),
                      help='''Which dataset in 'standard_data.py' to use''')
#data_arg.add_argument('--num_train_samples',type=int,
#                      help='''how many training samples to use''')


##--------------------    Model Args    ------------------------##
model_arg.add_argument('--load_model_file',type=str,default='',#unused at present
                     help='''where the actual checkpoint of model is
                             only useful if you start saving multiple models
                             per experiment--for example over training''')
model_arg.add_argument('--load_path',type=str,default='',
                      help='''load_path is typically some folder in ./logs''')
model_arg.add_argument('--model',type=str,default='',choices=dir(model_module),
                      help='''Which architecture in 'standard_model.py' to use''')


##--------------------    Misc Args    ------------------------##
#misc_arg.add_argument('--load_config',type=str2bool,default=False,  ##Now loads prev by default
#                      help='''whether to adopt the values from the previous model by default''')
misc_arg.add_argument('--is_train',type=str2bool,
                      help='''if we are training, what are we training?''')
#misc_arg.add_argument('--is_eval',type=str2bool,help='''are we evaluating''')
misc_arg.add_argument('--log_dir',type=str,default='./logs',
                     help='''where logs of model are kept''')
misc_arg.add_argument('--descrip',type=str,default='',
                     help='''a small string added to the end of the
                     model folder for future reference''')
misc_arg.add_argument('--prefix',type=str,default='Model',
                     help='''a small string added to the beginning of the
                     model folder for future reference''')


##--------------------    Training Args    ------------------------##

###Kind of hard to pick reasonable numbers for these across models
#train_arg.add_argument('--num_iter',type=int,default=3000,
#                       help='''number of steps to run the experiment for unless
#                       stopped by some other termination condition''')
#train_arg.add_argument('--log_every_n_steps'  ,type=int,default = 10,
#                       help='''how frequently to report to console''')
#train_arg.add_argument('--save_every_n_steps'  ,type=int,default = 1000,
#                       help='''how frequently to report save model''')
#train_arg.add_argument('--validation_every_n_steps' ,type=int,default = 2000,
#                       help='''how frequently to do evaluation''')
#train_arg.add_argument('--optimizer',type=str,choices=['grad_descent','adam']) #default=adam
train_arg.add_argument('--epochs',type=int,default=10,
                       help='''number of times to iterate through the dataset
                       dataset size determined by n_steps_per_epoch=
                       info['train'].num_examples//config.batch_size
                       ''')
train_arg.add_argument('--learning_rate',type=float,default=0.005)
#train_arg.add_argument('--learning_rate',type=float,default=0.0005)
#train_arg.add_argument('--dropout_keep_prob',type=float,default=1.0,
#                      help='''what fraction of inputs to keep''')


def load_config(info):
    if hasattr(info,'load_path'):
        load_path=info.load_path
    #elif isinstance(info,str):
    elif isinstance(info,six.string_types):
        if os.path.exists(info):
            load_path=info
        else:
            raise ValueError('I didnt plan for this')

    #loads config from previous model
    cf_path=os.path.join(load_path,'params.json')
    print('Attempting to load params from: ',cf_path)
    with open(cf_path,'r') as f:
        load_config=json.load(f)
    return load_config

def infer_model_file(cfx):
    #if 'model_file' not in cfx.keys():
    if not cfx.load_model_file:
        m_files=glob2.glob(cfx.load_path+'/checkp*/*')
        if len(m_files)!=1:
            raise ValueError('Expected 1 model file but got ',m_files)
        m_file=m_files[0]
        #cfx['load_model_file']=m_file
        cfx.load_model_file=m_file


def get_config():
    config, unparsed = parser.parse_known_args()

    #dataset_dir=os.path.join('data',config.dataset)
    #setattr(config, 'dataset_dir', dataset_dir )

    #if config.is_eval and not config.load_path:
    #    raise ValueError('must load path to evaluate')

    if len(unparsed)>0:
        print('WARNING, there were unparsed arguments:',unparsed)

    #these args change with each run
    #even if loading a previous model

    #dont_adopt=['model_dir','model_name','is_train','is_eval','load_path','load_config']
            #'data','descrip'

    if config.load_path:
        infer_model_file(config)#knows where these kinds of things are saved
        prev_config=load_config(config)#prev_config is dict while config is namespace
        pfx_prev_config={'previous_'+k:v for k,v in prev_config.items()}
        #pfx_prev_config={'previous_'+k:v for k,v in prev_config.__dict__items()}
        #pfx_prev_config={'load_'+k:v for k,v in prev_config.items()}

        config.__dict__.update(pfx_prev_config)

        if not config.data:
            config.data=prev_config['data']
        if not config.model:
            config.model=prev_config['model']# =config.previous_model

    return config,unparsed


###

#    if config.load_config:
#        print('WARNING: experimental loading of previous model config')
#        prev_config=load_config(config)
#        old_keys,update_keys=[],[]
#        for k in prev_config.keys():
#            if k not in dont_adopt:
#                if k not in config.__dict__.keys():
#                        old_keys.append(k)
#                elif config.__dict__[k]!=prev_config[k]:
#                        update_keys.append(k)
#        if len(old_keys)>0:
#            old_dict={k:prev_config[k] for k in old_keys}
#            print( "WARN: keys in previous model not in current model:",old_dict)
#            #config.__dict__.update(old_dict) #not unless old code is used#not implemented
#        if len(update_keys)>0:
#            update_dict={k:prev_config[k] for k in update_keys}
#            print("WARN: overwriting config with value from prev model:",update_dict)
#            config.__dict__.update(update_dict)
#
#    return config, unparsed


if __name__=='__main__':
    config,unparsed=get_config()


