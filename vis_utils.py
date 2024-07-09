import pandas as pd
import numpy as np
from config import get_config
from utils import prepare_dirs_and_logger,save_config
from tboard import file2number

import tensorflow as tf
import math
import numpy as np
import os
import sys
import glob2
from itertools import product
#sys.path.append('/home/chris/Software/workspace/models/slim')
from tqdm import trange
import time
import copy

from ArrayDict import ArrayDict

#temp for debug
from config import get_config
#from nonlinearities import name2nonlinearity


from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
This is a file to help visualize, reconstruct, what happened during network
training

'''

def standard_time():
    time = np.linspace(-0.2,0.4,150)
    zero_time=np.where(time>=0.)[0][0]# 0~time[50]
    return time, zero_time

#oldmthd# plt_X = X / X[:,zero_time:zero_time+1]#peak normalize
def volt_norm_peak(dataX):
    _,zero_time = standard_time()
    normX = dataX/dataX[:,zero_time:zero_time+1]
    return normX
def volt_norm1(dataX):
    '''(num x length-template) np.ndarray'''
    #seems to pickup on pk for both polarizations
    normX=dataX/np.max(np.abs(dataX),axis=1,keepdims=True)
    return normX

def ecg_format(ax):
    time,zero_time=standard_time()#const for all templates
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_yticklabels([])
    ax.set_ylim([-1.,1.])
    ax.set_xlim([time.min(),time.max()])

def get_figaxes(**kwargs):
#def get_figaxes(**kwargs, kw_return=True):
    ax = kwargs.pop('ax')  if 'ax'  in kwargs.keys() else None
    fig= kwargs.pop('fig') if 'fig' in kwargs.keys() else None
    figsize = kwargs.pop('figsize') if 'figsize' in kwargs.keys() else (8,6)
    if (fig and ax) is None:#"if either is None"
        fig,ax=plt.subplots(figsize=figsize)
        fig.set_tight_layout(True)
    #format#
    ecg_format(ax)
    return fig,ax
get_figaxes.kwfig=['ax','fig','figsize','func']


##Need to setup a whitelist for kwargs
def sgnsep_ecgs(pos_X,neg_X,**kwargs):
    time,zero_time=standard_time()#const for all templates

    if 'fig' in kwargs.keys():
        print('WARN: kwargs[\'fig\'] passed but not used')
        kwargs.pop('fig')
    if 'ax' in kwargs.keys():
        print('WARN: kwargs[\'ax\'] passed but not used')
        kwargs.pop('ax')

    pos_X=volt_norm1(pos_X)
    neg_X=volt_norm1(neg_X)

    #copy.deepcopy
    pos_fig,pos_ax=alpha_ecg(pos_X,**kwargs)#returns None,None if len(X)==0
    neg_fig,neg_ax=alpha_ecg(neg_X,**kwargs)#also does tight_layout

    return [pos_fig,neg_fig],[pos_ax,neg_ax]


def ovrlyd_ecgs(pos_X,neg_X,func=None,**kwargs):

    if min( len(pos_X), len(neg_X) )==0:
        return None,None

    time,zero_time=standard_time()#const for all templates
    fig,ax = get_figaxes(**kwargs)
    if len(pos_X)*len(neg_X)==0:
        raise ValueError('ecg plotter was passed empty data')

    minL=min(len(pos_X),len(neg_X))#should be same from resample_m anyway
    #params
    alP=40.#internal param
    alpha=float(alP)/(1.+float(minL))
    alpha=proj2interval(alpha,vmin=0.005)
    lnw = kwargs.pop('linewidth')  if 'linewidth'  in kwargs.keys() else 1.5
    #func = kwargs.pop('func')  if 'func'  in kwargs.keys() else None

    pos_X=volt_norm1(pos_X)
    neg_X=volt_norm1(neg_X)
    all_X=np.concatenate([pos_X,neg_X],axis=0)
    med_X=np.median( all_X ,axis=0)
    pos_med_X=np.median( pos_X ,axis=0)
    neg_med_X=np.median( neg_X ,axis=0)
    med_diff=np.max(np.abs( pos_med_X-neg_med_X ) )
    if med_diff>0.15:
        print('These two medians should be plotted seperately.\n',
              'diff is ',med_diff,
                '.. just gathing information')

    pos_c,neg_c='b','r'

    #def pos_plot(ax):
    def pos_plot():
        ax.plot(time,pos_X.T,c=pos_c,alpha=alpha,linewidth=lnw )
                #,**kwargs)
    def neg_plot():
        ax.plot(time,neg_X.T,c=neg_c,alpha=alpha,linewidth=lnw )
                #,**kwargs)

    ##There is some different dep. on order its plotted. 
    #   #arg.func or Op helps decide which is more important to emphasize
    #   ##interestingly order=+--+ or -++- may be a quick order-invariant hack 
    if func is None:
        order = '+-' #default order
    elif str(func)=='Min':#emphasize negatives of each node in lyr
        order = '+-'
    elif str(func)=='Max':
        order = '-+'
    else:
        raise ValueError('Thats it. Those are all the options. Game over.')

    for sgn in order:
        if sgn=='+':
            pos_plot()
        if sgn=='-':
            neg_plot()

    return fig,ax


def resample_m(Arr,m=4000,seed=22,return_idx=False):
    '''
    Arr    np array whose first dimension will be resampled
    m      many times.

    This is both used to add pad small and trim large ecg clumps
    '''

#    #assert len(Arr)>0
#    if len(Arr)==0:
#    if return_idx:
#        return Arr[row_i],np.
#    else:
#        return Arr[row_i]


    if len(Arr)==0:
        m=0  #np.choice returns array([], dtype=int64)

    #np.random.default_rng() is better approach but req. python 3
    np.random.seed(seed)
    row_i = np.random.choice(Arr.shape[0],m,replace=True)
    if return_idx:
        return Arr[row_i],row_i
    else:
        return Arr[row_i]


def plt_latex_settings():
    import matplotlib
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    #https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

    # font = {'family' : 'normal',
    #         'weight' : 'bold',
    #         'size'   : 22}

    # matplotlib.rc('font', **font)

def proj2interval(a, vmin=0.001, vmax=.95):
    return min(  max(a,vmin), vmax)

##-- ecg below
def alpha_ecg( X ,**kwargs):
    '''
    X is ecgs, [num_ecgs x len ecgs] shape
    alP is a constant scaling alpha
    alpha=a should be inversely proportional to len(X)=m many plots.
    We normalize to s/t the intensity const. of a pixel shaded by all m lines
    This contrib is m* [ a(1-a)^m]
    So setting a=1/m gives us a limit :) as m->inf
    alP just makes the colors workout in the right range
    '''

    #bailout
    if len(X)==0:
        return None,None
    time,zero_time=standard_time()#const for all templates

    #setup
    #print('alpha sees kw',kwargs.keys())
    fig,ax = get_figaxes(**kwargs)
    fig.set_tight_layout(True)

    #pproc
    plt_X = volt_norm_peak(X)
    medline=np.median( plt_X ,axis=0)

    #params
    alP=40.#internal param
    alpha=float(alP)/(1.+float(len(X)))
    alpha=proj2interval(alpha,vmin=0.005)
    linewidth=.5#mydefault

    kw_alpha={'alpha':alpha,
              'linewidth':linewidth,
              'c':'m',
             }
    keys_up = filter(lambda k:k in kw_alpha.keys(), kwargs.keys() )
    kw_alpha.update({k:kwargs[k] for k in keys_up})


    #linewidth = kwargs['linewidth'] if 'linewidth' in kwargs.keys() else linewidth
    #linewidth = kwargs.pop('linewidth') if 'linewidth' in kwargs.keys() else linewidth

    #plot
    ax.plot(time,plt_X.T, **kw_alpha )
    ax.plot(time,medline  ,c='b',linewidth=3.5)
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_yticklabels([])
    ax.set_ylim([-1.,1.])
    ax.set_xlim([time.min(),time.max()])

    return fig,ax





#def show_ecg_cluster( X ,fig=None,ax=None):
def show_ecgs( X ,fig=None,ax=None):
    if ax is None:
        fig,ax=plt.subplots()
        plt_latex_settings()#good enough
    time=np.linspace(-0.2,0.4,150)
    zero_time = np.where(time>=0.)[0][0]# 0~time[50]

    plt_X = X / X[:,zero_time:zero_time+1]#peak normalize

    medline=np.median( plt_X ,axis=0)
    ax.plot(time,plt_X.T,c='m')
    ax.plot(time,medline  ,c='b',linewidth=3.5)
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_yticklabels([])
    ax.set_ylim([-1.,1.])
    ax.set_xlim([time.min(),time.max()])
    return fig,ax


##-- ecg above

default_rng=[ [-3.55,-2.83],[1.78,1.89] ]
def sample_grid(rng=default_rng,res=200):
    lower=np.array(rng[0])
    upper=np.array(rng[1])
    sz=upper-lower
    #delta=0.005*np.min(sz)
    delta=np.min(sz)/float(res)
    x1=np.arange(lower[0],upper[0],delta)
    x2=np.arange(lower[1],upper[1],delta)
    GridX=np.stack(np.meshgrid(x1,x2),axis=-1)#LxLx2
    return GridX

def split_posneg(real_val,binary_val):
    pos_real=real_val[  np.where(binary_val==+1)[0] ]
    neg_real=real_val[  np.where(binary_val== 0)[0] ]
    pos_bin=binary_val[ np.where(binary_val==+1)[0] ]
    neg_bin=binary_val[ np.where(binary_val== 0)[0] ]
    return pos_real,neg_real,pos_bin,neg_bin


def subplots(*args,**kwargs):
    fig,axes=plt.subplots(*args,**kwargs)
    fig.tight_layout()

    try:
        iter_axes=iter(axes)
    except TypeError:
        #not iterable
        #print 'not iterable'
        iter_axes=[axes]
    else:
        #iterable
        #print 'iterable'
        pass
    for ax in iter_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_ticks_position('none') # tick markers
        ax.yaxis.set_ticks_position('none')
        ax.axis('off')

    return fig,axes

def paint_data(ax,npX,npY):
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)
    ax.plot(Xpos[:,0],Xpos[:,1],
             color='blue',
             linewidth=0.,
             marker='o',
             fillstyle='full',
             markeredgecolor='k',
    #          markeredgewidth=0.0)
            )
    ax.plot(Xneg[:,0],Xneg[:,1],
             color='red',
             linewidth=0.,
             marker='o',
             fillstyle='full',
             markeredgecolor='k',
            )

def paint_binary_contourf(ax,gX,eval_fn,thresh,cbar=False):
    eval_fn=np.array(eval_fn)
    if eval_fn.ndim==3:
        if eval_fn.shape[-1]==1:
            eval_fn=eval_fn[...,0]

    #lvls=[-0.0001+eval_fn.min(), thresh, eval_fn.max() +0.0001]
    lvls=[eval_fn.min(), thresh, eval_fn.max() ]
    gX0,gX1=gX[:,:,0],gX[:,:,1]
    ctf=ax.contourf(gX0,gX1,eval_fn,
                      levels=lvls,
                      cmap=plt.cm.bwr_r,
                      alpha=0.95,
                     )
    if cbar:
        plt.colorbar(ctf,ax=ax)
    return ctf

def infer_thresh(data):
    if np.min(data)<0:
        thresh=0.
    else:
        thresh=0.5
    return thresh
def draw_binary_contour(ax,gX,eval_fn,thresh=None):
    thresh=thresh or infer_thresh(eval_fn)

    eval_fn=np.array(eval_fn)
    if eval_fn.ndim==3:
        if eval_fn.shape[-1]==1:
            eval_fn=eval_fn[...,0]
    gX0,gX1=gX[:,:,0],gX[:,:,1]
    #clist=['g','c','m','y','k','b','r']
    #listLineStyle=['solid', 'dashed', 'dashdot', 'dotted']
    ctf=ax.contour(gX0,gX1,eval_fn,colors='g',levels=[thresh],linestyles='solid')
    return ctf

def surf(gX,eval_fn):
    eval_fn=np.array(eval_fn)
    if eval_fn.ndim==3:
        if eval_fn.shape[-1]==1:
            eval_fn=eval_fn[...,0]
    gX0,gX1=gX[:,:,0],gX[:,:,1]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(gX0, gX1, eval_fn,
           cmap=cm.coolwarm_r ,linewidth=0, antialiased=False)
    return ax,fig



def splitL(arr,keep_dims=False):
    '''
    convenience function for np.split into size 1 arrays along last axis
    '''
    szL=arr.shape[-1]
    split=np.split(arr,szL,axis=-1)
    if not keep_dims:
        split=[np.squeeze(A) for A in split]
    return split

####----------------old--_--------
#index [arch][dataset]
Pub_Model_Dirs=[[
    './logs/Pub_Model_0504_234437_D1A1',
    './logs/Pub_Model_0504_234557_D2A1',
    './logs/Pub_Model_0504_234822_D3A1'],
    [
    './logs/Pub_Model_0504_235012_D1A2',
    './logs/Pub_Model_0504_235058_D2A2',
    './logs/Pub_Model_0504_235237_D3A2'],
    [
    './logs/Pub_Model_0504_233305_D1A3',
    './logs/Pub_Model_0504_234152_D2A3',
    './logs/Pub_Model_0504_234319_D3A3'] ]







def get_path(name,prefix,log_dir):
    listglob=glob2.glob(log_dir+'/records/'+prefix+'*.npy')
    hasname=filter(lambda s:name in s[s.rfind(prefix):], listglob)
    if len(hasname)==1:
        return hasname[0]
    else:
        exactname='_'+name+'.'
        hasname=filter(lambda s:exactname in s[s.rfind(prefix):], listglob)
        if len(hasname)==1:
            return hasname[0]
        else:
            raise ValueError('Not exactly 1 match',[hn[hn.rfind('/'):] for hn in hasname])


#from nonlinearities import relu#np version
def get_np_network(weight_list):
    #weight_list is a list of pairs (weight,bias) starting from the input
    #network assumed to be relu
    def net(x):
        for W,b in weight_list:
            x=relu(b+np.dot(x,W))
        return x
    return net


def get_neuron_values(X,weight_list):
    #weight_list is a list of pairs (weight,bias) starting from the input
    #network assumed to be relu
    #returns value of each neuron on that grid

    PLayers=[]
    act=X
    for W,b in weight_list:
        h=b+np.dot(act,W)
        PLayers.append(h)
        act=relu( h )
    return PLayers


def vec_get_neuron_values(X,weights):
    '''
    This fn was meant to implement get_neuron_values but allow the weights to
    be arbitrary arrays whose slices were inputs xi, weight mats, and weight biases

    Does neuron computation but allows X and W to be indexed by the first
    arbitrary number of indicies, reserving the last 1 for the input dimension,
    the bias shape, or the last 2 for the weight shape.
    X has dim=xpad+1
    W has dim=wpad+2
    Args:
    X is an array of shape xindex.shape+(xdim,)
    weights is a list of weights,[W,b]
        W of shape widx.shape+(prev layer dim, next layer dim)
        b of shape widx.shape+(next layer dim)

    returns list, one per layer, of dim
    windex,xindex, layershape
    '''


    #X=GridX #try
    w0,b0=weights[0]
    wpad=len(w0.shape)-2
    sh_widx=w0.shape[:wpad]
    xpad=len(X.shape)-1

    #Reshaping
    Xshape=(1,)*wpad+X.shape+(1,)#last dim to multiply w
    rs_X=X.reshape(Xshape)

    def wrs(W):#also works on biases
        new_shape=sh_widx + (1,)*xpad + W.shape[wpad:]
        return W.reshape(new_shape)
    rs_weights=[[wrs(W),wrs(b)] for W,b in weights]

    #Network computation
    PLayers=[]
    act=rs_X
    for rs_W,rs_b in rs_weights:
        h=np.sum(act*rs_W,axis=-2)+rs_b#widx,xidx,layershape
        PLayers.append(h)
        act=np.expand_dims( relu( h ), -1 )
    return PLayers



def load_weights(log_dir):
    paths=[]
    ii=0
    while True:
        try:
            ii+=1
            W_str='W'+str(ii)
            b_str='b'+str(ii)
            W_pth=get_path(W_str,'wwatch',log_dir)
            b_pth=get_path(b_str,'wwatch',log_dir)
            paths.append([W_pth,b_pth])
            #print 'try_block',ii
        except:
            #print 'load weights fail','iter',ii
            #print 'except block',ii
            break
    #print paths #DEBUG
    weights=[[np.load(wp),np.load(bp)] for wp,bp in paths]
    return weights

def get_del_weights(weights):
    del_weights=copy.deepcopy(weights)#makes copy
    del_weights[-1][0]=del_weights[-1][0][...,1]-del_weights[-1][0][...,0]
    del_weights[-1][1]=del_weights[-1][1][...,1]-del_weights[-1][1][...,0]
    del_weights[-1][0]=np.expand_dims(del_weights[-1][0],axis=-1)#Leave 1 as last dim
    del_weights[-1][1]=np.expand_dims(del_weights[-1][1],axis=-1)
    return del_weights

def resample_grid(gridX,res=200):
    #better grid
    lower=gridX.reshape([-1,2]).min(axis=0)
    upper=gridX.reshape([-1,2]).max(axis=0)
    sz=upper-lower
    #delta=0.005*np.min(sz)
    delta=np.min(sz)/float(res)
    x1=np.arange(lower[0],upper[0],delta)#assuming 2D data
    x2=np.arange(lower[1],upper[1],delta)
    GridX=np.stack(np.meshgrid(x1,x2),axis=-1)#LxLx2
    return GridX

clist=['g','c','m','y','k','b','r']
listLineStyle=['solid', 'dashed', 'dashdot', 'dotted']



###Not used. was a mistake
def logdir_to_pthProb(log_dir,dt):
    record_dir=os.path.join(log_dir,'records')
    pth_Prob=os.path.join(record_dir,'Prob_dt'+str(dt)+'.npy')
    return pth_Prob



#    PLayers=vec_get_neuron_values(GridX,del_weights) #d*(TimexgX1xgX2xnl)




###The same code as in ipython but now it doesn't work...
#####CT_lists not defined on save() call
def generate_anim_file(log_dir):

    #log_dir=Pub_Model_Dirs[1][1]  #[arch#-1][data#-1]
    descrip,id_str=log_dir.split('_')[-1],str(file2number(log_dir))
    print 'using log_dir:',log_dir,' descrip:',descrip, 'id str:',id_str

    ##Load All
    record_dir=os.path.join(log_dir,'records')
    all_weights=load_weights(log_dir)
    all_step=np.load(get_path('step','wwatch',log_dir))#every 10 of 10000 iter

    ###Define Slicing###
    dt=10 #every 100
    iter_slice=np.arange(len(all_step))
    #iter_slice=np.linspace(0,100,2).astype('int')
    iter_slice=iter_slice[::dt]
    #iter_slice=iter_slice[-1:]#just last entry
    dt_weights=[[w[iter_slice],b[iter_slice]] for w,b in all_weights]
    #dt_weights=[[w[::dt],b[::dt]] for w,b in all_weights]
    del_weights=get_del_weights(dt_weights)
    #step=all_step[::dt]
    step=all_step[iter_slice]

    arch=[b.shape[-1] for w,b in dt_weights[:-1]]#net architecture
    gridX=np.load(get_path('gridX','hmwatch',log_dir))
    gX=resample_grid(gridX)#200  #standardized grid
    #HighResX=resample_grid(gridX,5000)
    npX=np.load(os.path.join(record_dir,'dataX.npy'))
    npY=np.load(os.path.join(record_dir,'dataY.npy'))
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)

    print '  calculating neuron values..'
    #d*(wpad,xpad,time,layersize)
    PLayers=vec_get_neuron_values(gX,del_weights) #d*(TimexgX1xgX2xnl)

    #linestyles=listLineStyle[l]
    lLS=listLineStyle[:-1]#Save dotted for boundary

    #colormap = plt.cm.gist_ncar
    #col_cycle=cycler('color', clist)
    def draw(time,layer,ax):
        ax.set_prop_cycle(plt.cycler('color', clist))
        contour_list=[]
        l=layer
        time_PLayers=[A[time] for A in PLayers]
        listL=splitL(time_PLayers[l])
        if layer+1<len(PLayers):
            ls=lLS[layer%len(lLS)]
        else:
            ls=listLineStyle[-1]
        for i,Pneu in enumerate(listL):
            ctf=ax.contour(gX0,gX1,Pneu,colors=clist[i%len(clist)],levels=[0.],linestyles=ls)
            contour_list.append(ctf)
        return contour_list


    ###ANIMATION###
    print '  beginning animation..'
    #fig,ax=plt.subplots(figsize=(14,12))
    #fig,axes=plt.subplots(2,2)
    fig,ax=plt.subplots()
    gX0,gX1=gX[:,:,0],gX[:,:,1]

    #global CT_lists#debug

    CT_lists=[]
    d=len(PLayers)-1
    for l in range(d+1):
        contour_list=draw(0,l,ax)
        CT_lists.append(contour_list)
    ax.scatter(Xpos[:,0],Xpos[:,1],marker='+',s=100,c='b',linewidth='3')
    ax.scatter(Xneg[:,0],Xneg[:,1],marker='_',s=100,c='r',linewidth='3')

    plt.tight_layout()

    plt.xticks([]) # labels 
    plt.yticks([])
    ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    def animate(time):
        #print 'time:',time
        #print 'ctl:',CT_lists
        global CT_lists#normally uncommented
        #print 'ctl:',CT_lists
        for contour_list in CT_lists:
            for ctf in contour_list:
                for c in ctf.collections:
                    c.remove()
        #global CT_lists#debug
        CT_lists=[]
        for l in range(d+1):
            CT_lists.append(draw(time,l,ax))
        return CT_lists

    #anim = animation.FuncAnimation(fig, animate, frames=len(step), repeat=True, interval=10)
    del_t=1
    anim_deep = animation.FuncAnimation(fig, animate, frames=np.arange(1,len(PLayers[0]),del_t), repeat=True,interval=10)
    #anim = animation.FuncAnimation(fig, animate, frames=np.arange(1,10000,del_t), repeat=True,interval=10)#, interval=500)

    fps=25
    Writer = animation.writers['ffmpeg']
    writer = Writer(metadata=dict(artist='Me'),fps=fps)
    anim_fname= (record_dir+'/'+id_str+'_overlay_'+'_fps'+str(fps)+
                   'start'+str(iter_slice[0])+
                    'end'+str(iter_slice[-1])+
                   'dur'+str(len(iter_slice))+
                   '.mp4')
    print '  saving file:',anim_fname
    anim_deep.save(anim_fname, writer=writer)
    print '  done.'



if __name__=='__main__':
    pass




