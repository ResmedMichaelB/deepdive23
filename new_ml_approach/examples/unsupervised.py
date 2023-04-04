import os, sys, glob, random, json
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append(r'C:\Users\MichaelB9\Git\deepdive23\new_ml_approach')

from helpers.load_data import *

batch_size=10
epochs=30

pacific_files=glob.glob(os.path.join(r'A:\dynamic\datasets\Pacific_ECS_Skellig_MVP\sessionGap_60mins','*','fg_sigs','*.json'))[::100]
random.Random(6).shuffle(pacific_files)
# Load the model
model=tf.keras.models.load_model(r'C:\Users\MichaelB9\Git\deepdive23\new_ml_approach\saved_models\position_model_09')
base_model=tf.keras.models.load_model(r'C:\Users\MichaelB9\Git\deepdive23\new_ml_approach\saved_models\position_model_09')
folder_path = r'C:\Users\MichaelB9\Data\S9 AHI\preprocessed'
target_signal='Position'
input_signals=['Flow', 'Mask Pres']

with open('train_subs.txt') as file:
    train_subs=file.read()
train_subs=train_subs.split('\n')

with open('val_subs.txt') as file:
    val_subs=file.read()
val_subs=val_subs.split('\n')

with open('test_subs.txt') as file:
    test_subs=file.read()
test_subs=test_subs.split('\n')



sig_tensor, target_tensor, subject_names = load_sig_tensor(
    val_subs[:-1],epoch_size = 30,epoch_step = 15,
    sample_rate = 25, input_signals=input_signals,
    target_signal=target_signal, #levels=5,
    dataset='s9',
    )

_, _, X_valF, y_val, _, _, _ = split(sig_tensor['Flow'],target_tensor,subject_names,train_size=0,val_size=1,test_size=0)
_, _, X_valP, _, _, _, _ = split(sig_tensor['Mask Pres'],target_tensor,subject_names,train_size=0,val_size=1,test_size=0)


sig_tensor=[]


for ep in range(epochs):


    # Load some new data
    for run in range(0,len(pacific_files),batch_size):

        print('__'*30)
        print('__'*30)
        print('Epoch {}. Batch {} of {}'.format(ep,run,len(pacific_files)))
        print('__'*30)
        print('__'*30)

        sig_tensor, target_tensor, subject_names = load_sig_tensor(
            train_subs[:-1],epoch_size = 30,epoch_step = 15,
            sample_rate = 25, input_signals=input_signals,
            target_signal=target_signal, #levels=5,
            dataset='s9',
            )

        X_trainF, y_train, _, _, _, _, _ = split(sig_tensor['Flow'],target_tensor,subject_names,train_size=1,val_size=0,test_size=0)
        X_trainP, _, _, _, _, _, _ = split(sig_tensor['Mask Pres'],target_tensor,subject_names,train_size=1,val_size=0,test_size=0)

        del sig_tensor

        cur_files=pacific_files[run:(run+batch_size)]

        ## Predict labels
        sig_tensor, target_tensor, subject_names = load_sig_tensor(
            cur_files,epoch_size = 30,epoch_step = 15,
            sample_rate = 25, input_signals=input_signals,
            target_signal='flow',#target_signal,
            dataset='pacific',ytype='Num'
            )

        # _, _, _, _, X_test1F,_, _ = split(sig_tensor['Flow'],target_tensor,subject_names,train_size=0,val_size=0,test_size=1)
        # _, _, _, _, X_test1P,_, _ = split(sig_tensor['Mask Pres'],target_tensor,subject_names,train_size=0,val_size=0,test_size=1)
        
        X_train1F, _, _, _, _, _, _ = split(sig_tensor['Flow'],target_tensor,subject_names,train_size=1,val_size=0,test_size=0)
        X_train1P, _, _, _, _, _, _ = split(sig_tensor['Mask Pres'],target_tensor,subject_names,train_size=1,val_size=0,test_size=0)

        y_train1=base_model.predict([X_train1F,X_train1P])
        y_train1=np.where(y_train1>0.5,1,0)

        X_trainF=np.concatenate([X_trainF,X_train1F],axis=0)
        X_trainP=np.concatenate([X_trainP,X_train1P],axis=0)
        y_train=np.concatenate([y_train,y_train1.reshape(-1)])

        del X_train1F
        del X_train1P

        # Train the model on new labels
        H=model.fit(
            [X_trainF, X_trainP],y_train,
            epochs=10,batch_size=64,
            shuffle=True,validation_data=([X_valF, X_valP],y_val),
            )

        del X_trainF
        del X_trainP

    model.save(r'C:\Users\MichaelB9\Git\deepdive23\new_ml_approach\saved_models\position_model_10')
    # Back to step 2

    # Load the validation data. Evaluate

    # Load the train data. Train the model

    # with open(r'C:\Users\MichaelB9\Data\S9 AHI\Tensors\Model9\train.json') as file:
    #     encoded=json.load(file)
    #     train_data=np.array(json.loads(encoded))
