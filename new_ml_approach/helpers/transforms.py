import numpy as np
from scipy import signal


def list2numpy(data):
    for sig in data['fs']:
        if type(data[sig])==list:
            data[sig]=np.array(data[sig])
    
    return data

def numpy2list(data):
    for sig in data['fs']:
        if type(data[sig])==np.ndarray:
            data[sig]=list(data[sig])
        
        data['fs'][sig]=float(data['fs'][sig])
            
    return data

def filterLP(y,fs,lp_cutoff,filt_order):
    b,a= signal.butter(filt_order,
                        lp_cutoff,
                        btype='low',
                        analog=False,
                        fs=fs)
    return signal.filtfilt(b,a,y,axis=0)


def get_octave(sig,fs,levels,preQuantiles=[]):

    new_sig=[]
    quantiles=[]
    low=sig
    split=fs//2
    i=0
    while levels>0:
        low=filterLP(low,fs,split,4)
        high=sig-low
        if len(preQuantiles)==0:
            quantiles.append(np.quantile(high,0.95))
            new_sig.append(np.sign(high)*np.log(np.abs(high)/np.quantile(high,0.95)+1))
        else:
            new_sig.append(np.sign(high)*np.log(np.abs(high)/preQuantiles[i]+1))

        split=split/2
        levels=levels-1
        i=i+1
    
    if len(preQuantiles)==0:
        quantiles.append(np.quantile(low,0.95))
        new_sig.append(np.sign(low)*np.log(np.abs(low)/np.quantile(low,0.95)+1))
    else:
        new_sig.append(np.sign(low)*np.log(np.abs(low)/preQuantiles[i]+1))


    return new_sig, quantiles

def reformat_fs(data):

    data['fs']={}
    for sig in data.keys():
        if sig + '_fs' in data['metadata']:
            data['fs'][sig]=data['metadata'][sig + '_fs']
    
    return data