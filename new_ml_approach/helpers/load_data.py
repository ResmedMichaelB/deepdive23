
import numpy as np
import json, os, sys, random
import scipy as sp
import pandas as pd

cur_path=os.getcwd()
sys.path.append(cur_path)

parent_path=os.path.abspath(os.path.join(cur_path,'..'))
sys.path.append(parent_path)

from helpers.transforms import get_octave, list2numpy, reformat_fs
from helpers.config import *


def load_sig_tensor(json_filenames, epoch_size, epoch_step, sample_rate, input_signals, target_signal ,ytype = 'Cat', dataset = 's9'):
    
    '''
    Function that loads data from json files. Resamples signals to match the set sample rate
    Data is then transformed through octave encoding and reshape into tensor (octave chans x epochs x time points) 
    ready to be input to a 1d cnn model.

    Function takes desired input signals each a 'head' for a multihead model.
    A target variable is also set. 
    '''

    # Create empty dictionary with each key pair the extracted signal and its reshaped tensor
    sig_tensor = {sig:[] for sig in input_signals}
    # Subject name tracks each row of the tensor by subject name. Used for splitting the data
    subject_name = {sig:[] for sig in input_signals}
    target_tensor=[]

    for fname in json_filenames:
        
        subject_id = fname.strip('.json').split('\\')[-1]
        # print('Processing ', subject_id)
        
        # Open json file
        with open(fname) as json_file:
            data = json.load(json_file)

        if 'sessions' in data.keys():
            if 'ecg_processed' in data.keys():
                data['sessions'][0]['heartrate']=data['ecg_processed']['processed_ecg']['hr']
                data['sessions'][0]['metadata']['heartrate_fs']=1.5
            
            data = data['sessions'][0]
        
        if 'metadata' in data:
            data = reformat_fs(data)

        data = list2numpy(data)

        # Label position consistently
        if dataset == 's9':
            data['Position'] = np.array(list(map(position_map,data['Angle'])))
            data['fs']['Position'] = data['fs']['Angle']
        elif dataset == 'fillius':
            data['position'] = np.array(list(map(fillius_map,data['position'])))
            data['psg_hyp'] = np.array(list(map(sleep_stage_map,data['psg_hyp'])))
            data['fs']['position'] = 16


        # Find shortest signal and truncate data
        minL = np.min([len(data[L])/data['fs'][L] for L in input_signals])

        for sig in input_signals:
            
            # Truncate signal to the minimum length
            data[sig] = data[sig][0:int(minL*data['fs'][sig])]
            # Resample the data
            if data['fs'][sig] != sample_rate:
                sp.signal.resample(data[sig],len(data[sig])/data['fs'][sig])
            
            # Octave encoding, transform 1d signal into multiple frequnecy bands
            all_sigs,_ = get_octave(sig = data[sig],fs = data['fs'][sig],levels = 7)
            

            # all_sigs is a 2d matrix (Octave channels x time points)
            all_sigs = np.stack(all_sigs)
            # Truncate the portion of the signal that doesn't fit into a full window
            if (all_sigs.shape[1]%(epoch_size*sample_rate))>0:
                all_sigs = all_sigs[:,0:-(all_sigs.shape[1]%(epoch_size*sample_rate))]
            
            # Epoch the signal. Becomes a 3d tensor (octaves x number of epochs x time points)
            epoch_sig = all_sigs.reshape(all_sigs.shape[0],int(all_sigs.shape[1]/(epoch_size*sample_rate)),int(epoch_size*sample_rate))
            
            # Slot in steps if its a sliding window with overlap
            if epoch_step==epoch_size/2:
                all_sigs_shift = np.roll(all_sigs,-epoch_step*sample_rate,axis = 1)
                epoch_sig_shift = all_sigs_shift.reshape(all_sigs.shape[0],int(all_sigs.shape[1]/(epoch_size*sample_rate)),int(epoch_size*sample_rate))
                epoch_sig_combo = np.concatenate([epoch_sig,epoch_sig_shift],axis = 2)
                epoch_sig_combo = epoch_sig_combo.reshape((epoch_sig_combo.shape[0],epoch_sig_combo.shape[1]*2,int(epoch_sig_combo.shape[2]/2)))
                epoch_sig = epoch_sig_combo

            sig_tensor[sig].append(epoch_sig)
            subject_name[sig].append([subject_id]*epoch_sig.shape[1])


        data[target_signal]=data[target_signal][0:int(minL*data['fs'][target_signal])]
        target_epoch_length=int(epoch_size*data['fs'][target_signal])
        
        if (len(data[target_signal])%(target_epoch_length))>0:
            data[target_signal] = data[target_signal][0:-(len(data[target_signal])%(target_epoch_length))]
        
        target=[]
        
        for i in np.arange(0,len(data[target_signal]),int(epoch_step*data['fs'][target_signal])):
            
            if ytype == 'Cat':
                target.append(sp.stats.mode(data[target_signal][i:i+target_epoch_length],keepdims=False)[0])
            else:
                target.append(np.mean(data[target_signal][i:i+target_epoch_length]))

        target_tensor.append(np.array(target))


    for key in sig_tensor:

        sig_tensor[key] = np.concatenate(sig_tensor[key],axis = 1)
    
    subject_name = np.concatenate(subject_name[list(subject_name.keys())[0]])
    target_tensor = np.concatenate(target_tensor)

    if target_signal.lower() == 'position':
        target_tensor=np.where(target_tensor==0,1,0)
    
    # if target_signal.lower() == 'psg_hyp':
    #     target_tensor=np.where(target_tensor==1,1,0)

    
    return sig_tensor, target_tensor, subject_name




def split(X,y,subject_names,train_size=0.8,val_size=0.1,test_size=0.1):

    '''
    Data split function. Works like train test split except data is separated by subject
    '''

    subject_ids=list(set(subject_names))
    random.Random(20).shuffle(subject_ids)

    train_subs=subject_ids[0:int(len(subject_ids)*train_size)]
    val_subs=subject_ids[int(len(subject_ids)*train_size):(int(len(subject_ids)*train_size)+int(len(subject_ids)*val_size))]
    test_subs=subject_ids[(int(len(subject_ids)*train_size)+int(len(subject_ids)*val_size)):]

    print('-'*30)
    print('Train-test-validation split')
    print('-'*30)

    print('Training Set Size = ', len(train_subs), ' subjects')
    print('Validation Set Size = ', len(val_subs), ' subjects')
    print('Test Set Size = ', len(test_subs), ' subjects')

    X_train = X[:,np.where(np.isin(subject_names,train_subs))[0],:]
    X_train = np.transpose(X_train, (1,2,0))
    y_train = y[np.where(np.isin(subject_names,train_subs))[0]].reshape(-1)
        
    # Shuffle the data (helps with batch grad descent)
    idx=np.arange(X_train.shape[0])
    random.Random(6).shuffle(idx)
    X_train=X_train[idx,:,:]
    y_train=y_train[idx]

    X_val=X[:,np.where(np.isin(subject_names,val_subs))[0],:]
    X_val = np.transpose(X_val, (1,2,0))

    y_val = y[np.where(np.isin(subject_names,val_subs))[0]].reshape(-1)

    X_test=X[:,np.where(np.isin(subject_names,test_subs))[0],:]
    X_test = np.transpose(X_test, (1,2,0))
    y_test = y[np.where(np.isin(subject_names,test_subs))[0]].reshape(-1)

    print('-'*30)
    print('train set size = ', X_train.shape, '\n Target distribution \n',pd.DataFrame(pd.Series(y_train).value_counts(normalize=True)))
    print('-'*30)
    print('val set size = ', X_val.shape, '\n Target distribution \n',pd.DataFrame(pd.Series(y_val).value_counts(normalize=True)))
    print('-'*30)
    print('test set size = ', X_test.shape, '\n Target distribution \n',pd.DataFrame(pd.Series(y_test).value_counts(normalize=True)))
    print('-'*30)


    return X_train, y_train, X_val, y_val, X_test, y_test