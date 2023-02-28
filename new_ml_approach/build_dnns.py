from keras.models import Sequential, Model
from keras.layers import *
# from keras.layers.merge import Concatenate
import tensorflow as tf
# from tensorflow.keras import backend as K
import numpy as np



def build_cnn_1D(n_timesteps, n_features,filters=16,kernel_size=128,n_dense=32,dropout=0.5,ytype='Cat'):

    '''
    1D CNN model with 2 CNN layers and 2 dense layers.
    Dropout included for regularisation
    '''

    model = Sequential()

    # Layer 1
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(n_timesteps,n_features)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))

    # Layer 2. Number of filters doubles
    model.add(Conv1D(filters=filters*2, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))

    # Layer 3. Flatten and dense
    model.add(Flatten())
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dense(n_dense//2, activation='relu'))
    
    if ytype=='Cat':
        model.add(Dense(1,activation='sigmoid'))
    else:
        model.add(Dense(1,activation='linear',use_bias=False))

    return model