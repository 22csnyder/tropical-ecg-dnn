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

from config import get_config
from utils import prepare_dirs_and_logger,save_config,make_folders
from utils import relu,fg_diff,hinge
from utils import batch_inner,batch_outer
from utils import outer_inner,cwise_inner
from utils import SH,DIFF,EQUAL #db handy
from utils import logit_bxe #cross entropy helper
from utils import f_minus_g

import standard_data
import standard_model
from standard_data import peak,inspect_dataset,tuple_splits
from standard_model import tropical_objects #help load custom layers

from ArrayDict import ArrayDict
import time##db


from tropical_layers import TropicalRational

'''
Load tropical saved model.
Use greedy method to approx
Evaluate & Save greedy approximation model
'''





if __name__=='__main__':
    plt.close('all')

#-------------Config and Log Handling--------------#
    config,_=get_config()#arg parsing
    prepare_dirs_and_logger(config)
    save_config(config)

    # ~ ~ Encouraged to use these folder names ~ ~ #
    model_dir=config.model_dir
    bhsz= config.batch_size
    checkpoint_dir=os.path.join(model_dir,'checkpoints')
    make_folders([checkpoint_dir])#,summary_dir])

    #if not config.load_model_file:
    config.model_file=os.path.join(checkpoint_dir,'Model_ckpt.h5')
    model_file=config.model_file
    save_config(config)
    print('[*] Model File:',model_file)


#-------------Begin Data--------------#
    datasets,info=getattr(standard_data,config.data)()

    ds_train=datasets['train']
    ds_input=ds_train.map(lambda e:e['x']).batch(1)


    #modify if taking subset
    try:
        train_size=info.train_size
        test_size=info.test_size
    except:
        train_size=info.splits['train'].num_examples
        test_size=info.splits['test'].num_examples
    bhsz      =config.batch_size
    train_tuple,test_tuple=tuple_splits(datasets)
    bh_eval_data=test_tuple.batch(bhsz)

    #---reference batch for datasets---#
    db_bat=peak(ds_train.batch(25))
    X25=db_bat['x']
    X20=X25[:20]
    X=X20

#-------------Begin Model--------------#
    #u'./logs/Model_1114_164617_dbTropSave/checkpoints/Model_ckpt.h5'
    if config.load_path:
        #loads old model for new experiments
        #old model_dir/ folder will be unchanged

        #model=keras.models.load_model(config.load_path)
        model=keras.models.load_model(config.load_model_file,
            custom_objects=tropical_objects)

        #maybe save model in new folder?
    else:
        raise Exception('First train a model with main.py')
        ##New model
        #model_get=getattr(standard_model,config.model)
        #model=model_get(input_shape=info.input_shape)


    model.evaluate(bh_eval_data)
    net_x,trop_x,ll_x=model.validate_tropical(X)

    fn_compare=np.concatenate([net_x,trop_x,ll_x],-1)
    print('fn compare:')
    print('net(x)  trop(x)  ll(x)')
    print(fn_compare)









#    ##I dont fully understand when keras.Model will work
#
#    def tr(x):
#        fg_x = [x,tf.zeros_like(x)]   # or not?
#        for L in model.layers:
#            fg_x=L.fg_layer(fg_x)
#        return fg_diff(  fg_x  )
#
#    fg_x = [X,tf.zeros_like(X)]   # or not?
#    for L in model.layers:
#        fg_x=L.fg_layer(fg_x)
#    tr_x=fg_diff( fg_x )
#    #return fg_diff(  fg_x  )
#
#    def n(x):
#        l_x=x
#        for L in model.layers:
#            l_x=L(l_x)
#        return l_x
#    M3=keras.Model(inputs=model.input,outputs=n(model.input))  #yes
#    tropModel=keras.Model(inputs=model.input,outputs=tr(model.input)) #no
#
#    M3.compile(model.optimizer, model.loss, model.metrics)
#    tropModel.compile(model.optimizer, model.loss, model.metrics)
#
#   M3.evaluate(bh_eval_data)   #fine
#   tropModel.evaluate(bh_eval_data) #not fine




### ---------- Greedy Algorithm ---------- ###
    ##cfx##
    #n_greedy_iter=500
    #n_greedy_iter=50
    #n_greedy_iter=15
    n_greedy_iter=15
    #n_greedy_iter=1
    #Train_Data=datasets['train'].batch(32).repeat()
    Train_Data=datasets['train'].batch(256).repeat()
    greedy_data=Train_Data.take(n_greedy_iter)
    #net=model
    #fg_model=trop_model

    #db
    #greedy_data=greedy_data.skip(n_greedy_iter-1)


    ##cfx##

    ##Notes##
        #Throughout, I try to use caps F,G,Poly, 
        # for var names instead of f,g,poly when referring to current 
        # accumulated model pred, and lowercase for
        # preds based on local lin approx to current batch
        #"sig sample iter"--when was this guy added
        #    keys have to be lower for both

        #rule for pmy_polys "plus/minus y" 
        #"my"=-y_pm1  "py"=+y_pm1
        #for my_poly:
        #    y>0 eval g_sig(x)(x), for y<=0 eval f_sig(x)(x)
        # reverse for py_poly

    #db
    #bh = peak( greedy_data )

    ###   Init SS Loop   ###
    ss_iter = -1
    Fw = tf.zeros( (1,)+model.input.shape[1:],dtype=model.input.dtype)
    Fb = tf.zeros( (1,1), dtype=model.output.dtype )
    Gw = tf.zeros( (1,)+model.input.shape[1:],dtype=model.input.dtype)
    Gb = tf.zeros( (1,1), dtype=model.output.dtype )

    hat_fg_kernel = tf.zeros( (1,2)+model.input.shape[1:],dtype=model.input.dtype)
    hat_fg_bias   = tf.zeros( (1,2)                      ,dtype=model.input.dtype)
    ad_Greed=ArrayDict({
        'kernel':hat_fg_kernel,
        'bias'  :hat_fg_bias,
        'delete me' :True, #remove zero fn?
    })
    ###   Init SS Loop   ###

    ad_Packrat=ArrayDict()

    print('Starting greedy loop with ',n_greedy_iter,' iterations')
    t0=time.time()
    for ss_iter, bh in enumerate(greedy_data):

        #if ss_iter>0:
        #    break
        #print('ss_iter',ss_iter)

        x,y=bh['x'],bh['y'] #actual label
        ny =tf.reshape( model.predict_classes(x), y.shape ) #net label
        bh['ny']=ny

        y_hot = tf.one_hot(y,depth=2)#(bsz,2)#  has a 1 in first col if y=0
        y_pm1 = tf.reshape( tf.cast(y,'float')*2 - 1, [-1,1])
        ny_hot = tf.one_hot(ny,depth=2)#(bsz,2)#  has a 1 in first col if y=0
        ny_pm1 = tf.reshape( tf.cast(ny,'float')*2 - 1, [-1,1])#bsz x 1

        bh_ones=np.ones_like(y)#each sample keeps track of the stats of the batch its in
        hat_fg_kernel = ad_Greed['kernel']#linear fns so far
        hat_fg_bias   = ad_Greed['bias'  ]

        #fg_x,DfDg_x=eval_f_and_Df( fg_model,  bh['x'] )
        fg_x,fg_Linear,fg_Bias = model.fg_call( bh['x'] )
        bh_fgx      =tf.concat(fg_x,1)    #bhsz ,2
        bh_kernel   =tf.stack(fg_Linear,1)#bhsz , 2,  x.shape
        bh_bias     =tf.concat(fg_Bias,1) #bhsz , 2
        bh['kernel']=bh_kernel#linear fns at x
        bh['bias'  ]=bh_bias
        bh['ss_iter']=ss_iter * bh_ones

        outer_hat_kernel_x=outer_inner(x,hat_fg_kernel)+hat_fg_bias#bhsz x k_filters x 2
        hat_fg_x=tf.reduce_max(outer_hat_kernel_x,axis=1) # bhsz x 2
        hat_model_x=hat_fg_x[:,:1]-hat_fg_x[:,1:]

        #new section
        gy_fg_x = tf.reduce_max(outer_hat_kernel_x,axis=1,keepdims=True) # bhsz x 1 x 2
        #gy_x    = f_minus_g( tf.squeeze(gy_fg_x) ) #bsz x 1
        gy_x    = f_minus_g( gy_fg_x ) #bsz x 1 x 1 
        bh_table= outer_inner( x, bh_kernel )+bh_bias #bsz x bsz(filters) x 2
        up_gy_fg_x=tf.maximum(bh_table,gy_fg_x)#up=update #bsz x (bsz filters) x 2
        up_gy_x   = f_minus_g(up_gy_fg_x) #bsz x bsz x 1
        #up_gy_x   = tf.squeeze( up_gy_x ) #bsz x bsz

        ed_ny_pm1=tf.expand_dims(ny_pm1,-1)
        prev_gy_margin = ed_ny_pm1 * gy_x  #bsz x 1 x 1
        next_gy_margin = ed_ny_pm1 * up_gy_x #bsz x (bsz filters) x1

        ##New##
        sample_loss=lambda T: relu(-T)
        prev_sloss = sample_loss( prev_gy_margin )
        next_sloss = sample_loss( next_gy_margin )#bsz x bsz_filters x 1

        bh_forwards_progress  = relu(  prev_sloss - next_sloss  )
        bh_backwards_progress = relu( -prev_sloss + next_sloss  )#bsz x bsz x 1
        frac_improved      = np.mean(bh_forwards_progress>0.,axis=0)
        forwards_progress  = np.mean( bh_forwards_progress,axis=0)
        backwards_progress = np.mean(bh_backwards_progress,axis=0)#bsz x 1

        batch_loss=lambda T:tf.reduce_mean( relu(-T),axis=0 )#batch<->axis=0
        prev_gy_loss=tf.squeeze( batch_loss( prev_gy_margin )) # scalar
        next_gy_loss=            batch_loss( next_gy_margin )  #bsz filters x 1

        loss_decrease = prev_gy_loss-tf.reduce_min(next_gy_loss)#scalar

        bh_gfx = tf.reverse(bh_fgx     ,[-1])#switch columns<->reflect left-right
        gy_gfx = tf.reverse(hat_model_x,[-1])

        #Margins for update
        pm_margins = hat_fg_x - bh_gfx#  [hatf(x)-g(x)  , hatg(x)-f(x)]
                #for y=0, want fhat>g (pos firstcol)
        margin=tf.reshape(tf.boolean_mask(pm_margins,ny_hot),[-1,1])
        gy_margin = ny_pm1 * hat_model_x
        #margin=tf.reshape(tf.boolean_mask(pm_margins,y_hot),[-1,1])
        #gy_margin = y_pm1 * hat_model_x

    ####   Define Loss & Sample Choice  ####
        #bh_loss  = -margin # -> many oscillations, poor convergence
        #bh_loss  = -gy_margin
        #bh_loss  = batch_loss( gy_margin )
        bh_loss   = -gy_margin
        bh_informative = prev_gy_loss - next_gy_loss

        #get_ind=np.argmax(bh_informative)
        get_ind=np.argmax(forwards_progress)#ignore estimates made worse
        #get_ind=np.argmax(frac_improved)
    ####   Define Loss & Sample Choice  ####

        bh['bh margin']=margin
        bh['forwards_progress' ]=forwards_progress#bsz x 1
        bh['backwards_progress']=backwards_progress
        bh['frac_improved']=frac_improved


    ##Collect Data
        bh['frac no new info']=np.mean(bh_informative==0.)*bh_ones
        bh['prev_gy_loss']=prev_gy_loss * bh_ones
        bh['loss_decrease']=loss_decrease * bh_ones
        bh['bh informative']=bh_informative

        m_margins_improved = np.sum(next_gy_margin>prev_gy_margin,axis=0)
        bh['m_margins_improved']=m_margins_improved
        bh['bestchoice_m_margins_improved']=np.max(m_margins_improved)*bh_ones

        #model statistics
        bh['gy margin']    =gy_margin #how well did the model do
        bh['gy acc']       =np.mean(gy_margin>0.) * bh_ones
        bh['ave gy margin']=np.mean(gy_margin) * bh_ones
        #batch statisitics
        bh['net(x)']       = model( x )
        bh['bh loss']      =bh_loss
        bh['ave bh loss']  =np.mean(bh_loss)  *bh_ones
        bh['bh acc']       =np.mean( bh_loss>0)    *bh_ones
        bh['total time']   =(time.time()-t0)*bh_ones


        ad_Batch=ArrayDict(bh)#just easier to slice
        new_info=ad_Batch[ get_ind:get_ind+1: ]#slice(n,n+1)# keeps batchdim of bsz=1

        if 'delete me' in ad_Greed.keys():#throw away 0 as starting pt
            assert ss_iter==0
            ad_Greed=new_info
            bh0=ad_Batch
        else:#ss_iter>0
            assert ss_iter>0
            ad_Greed.concat(new_info)

        ad_Packrat.concat(ad_Batch)#just keep everything

    print('Finished ',1+ss_iter,' steps of Greedy Sig in ',time.time()-t0)

    adG=ad_Greed
    ad_ny=adG['ny']
    ad_ny_pm1 = tf.reshape( tf.cast(ad_ny,'float')*2 - 1, [-1,1])#bsz x 1
    ad_x,ad_y = ad_Greed['x'],ad_Greed['y']
    ad_fx, ad_gx= model.fg_call( tf.constant(ad_x) )[0]
    ad_pix=np.where(ad_ny==1)[0]
    ad_nix=np.where(ad_ny==0)[0]
    adxp,adxn=ad_x[ad_pix],ad_x[ad_nix]
    pos_ad_fx,pos_ad_gx=model.fg_call( tf.constant(adxp) )[0]
    neg_ad_fx,neg_ad_gx=model.fg_call( tf.constant(adxn) )[0]


    #sanity check
    ad_kernel,ad_bias=ad_Greed['kernel'],ad_Greed['bias']
    #These preds should all output the same thing
    net_pred=model(ad_x)
    ad_fx, ad_gx= model.fg_call( tf.constant(ad_x) )[0]
    fgcall_pred=f_minus_g( model.fg_call( tf.constant(ad_x) )[0] )

    ed_ad_x=tf.expand_dims(ad_x,1)

    #ad_inner = cwise_inner(ad_x,ad_kernel)
    ad_inner = cwise_inner(ed_ad_x,ad_kernel) + ad_bias
    ad_ipred =f_minus_g( ad_inner )

    ad_outer = outer_inner( ad_x,ad_kernel ) + ad_bias
    ad_amax  = np.argmax( ad_outer, axis=1 ) # should give range(1,ss_iter)
    ad_ofg   = np.max(    ad_outer, axis=1 )
    ad_opred = f_minus_g( ad_ofg )

    Preds=np.concatenate([ net_pred,fgcall_pred, ad_ipred, ad_opred, ad_y, ],axis=1)


    #########

    trf=TropicalRational(kernel=ad_kernel,bias=ad_bias,input_shape=info.input_shape)
    trf_model=keras.models.Model(inputs=model.inputs,outputs=trf(model.input))

    #trf_model.compile(model.optimizer,model.loss,model.metrics)
    ####object of type 'NoneType' has no len()
    # 343       endpoint.create_training_target(t,
                                                # run_eagerly=self.run_eagerly)

    #########

    plt.scatter(adG['gy margin'],adG['bh margin'])
    plt.xlabel('greedy model margin')
    plt.ylabel('update criteria margin')

    plt.figure()
    iters=adG['ss_iter']
    plt.plot(iters,adG['ave bh loss'])
    plt.xlabel('iters')
    #plt.ylabel('batch accuracy')
    plt.ylabel('batch loss')

    plt.figure()
    plt.plot(iters,adG['gy acc'])
    plt.ylabel('greedy model train(batch) accuracy')
    plt.xlabel('iters')

    #plt.figure()
    #plt.scatter(ad_fx,ad_gx)
    #plt.xlabel('f(x)')
    #plt.ylabel('g(x)')

    plt.figure()
    plt.scatter(pos_ad_fx,pos_ad_gx,c='b')
    plt.scatter(neg_ad_fx,neg_ad_gx,c='r')
    plt.xlabel('f(x)')
    plt.ylabel('g(x)')

    plt.figure()
    plt.scatter(bh_loss,bh_informative)
    plt.xlabel('-margin current model on sample')
    plt.ylabel('dec. in model batch loss due to adding sample')
    plt.title('helpful vs informative on last batch')

    plt.figure()
    plt.plot( adG['bh loss'], adG['bh informative'], '-o')
    plt.xlabel('-margin chosen samples')
    plt.ylabel('dec. in model batch loss due to adding sample')
    plt.title('helpful vs surprise throughout iteration')

    plt.figure()
    plt.plot(iters,adG['frac no new info'])
    plt.title('fraction uninformative for whole batch')
    plt.ylim([0,1.])
    plt.xlabel('iters')

    plt.figure()
    plt.plot(iters,adG['frac_improved'])
    plt.title('fraction of the batch improved at each step')
    plt.ylim([0,1.])
    plt.xlabel('iters')

    upper_fp=np.max(forwards_progress)
    plt.figure()
    plt.ylabel('dec. in model batch loss due to adding sample')
    #plt.xlabel('batch-ave relu(sample loss decrease)')
    plt.xlabel('total positive improvement(div by'+str(np.round(upper_fp))+') or frac_improved')
    plt.title('last batch metrics for informativeness')
    plt.scatter(forwards_progress/upper_fp,bh_informative,c='b')
    plt.scatter(frac_improved,bh_informative,c='r')


    #plt.show(block=False)
    #plt.show()

    plt.close('all')



#check over all accuracy
#XX=np.unique(ad_Packrat['x'],axis=0)
#TT=f_minus_g(  np.max( outer_inner(XX, ad_kernel)+ad_bias,axis=1) ) 
#NN=model(XX)
#TvN=np.concatenate([TT,NN],axis=-1)
#print('acc=',np.mean( np.sign(TT)==np.sign(NN) ))


