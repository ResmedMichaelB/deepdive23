import os, sys, glob, random, json
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append(r'C:\Users\MichaelB9\Git\deepdive23\new_ml_approach')

from helpers.plotting import train_val_curve, plot_roc
from sklearn.metrics import *
from helpers.load_data import *

with open('test_subs.txt') as file:
    test_subs=file.read()
test_subs=test_subs.split('\n')

model=tf.keras.models.load_model(r'C:\Users\MichaelB9\Git\deepdive23\new_ml_approach\saved_models\position_model_09')

target_signal='Position'
input_signals=['Flow', 'Mask Pres']

print('Loading Data...')
sig_tensor, target_tensor, subject_names = load_sig_tensor(
    test_subs[:-1],epoch_size = 30,epoch_step = 15,
    sample_rate = 25, input_signals=input_signals,
    target_signal=target_signal, #levels=5,
    dataset='s9',
    )

_, _, _, _, X_testF, y_test, _ = split(sig_tensor['Flow'],target_tensor,subject_names,train_size=0,val_size=0,test_size=1)
_, _, _, _, X_testP, _, _ = split(sig_tensor['Mask Pres'],target_tensor,subject_names,train_size=0,val_size=0,test_size=1)


y_pred=model.predict([X_testF,X_testP])

plt.figure(figsize=(3,3))

plot_roc(y_test,y_pred,0.5, label='Test set')
print('here')