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



def build_cnn_1D_5layer(n_timesteps, n_features,filters=4,kernel_size=5,n_dense=32,dropout=0.5,ytype='Cat'):

    '''
    1D CNN model with 2 CNN layers and 2 dense layers.
    Dropout included for regularisation
    '''

    model = Sequential()

    # Layer 1
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=kernel_size, input_shape=(n_timesteps,n_features)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    # model.add(MaxPooling1D(pool_size=2))

    # Layer 2. 
    model.add(Conv1D(filters=filters, strides=kernel_size//2, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))
    
    # Layer 3. Number of filters doubles
    model.add(Conv1D(filters=filters*2, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))
    
    # Layer 4. Number of filters doubles
    model.add(Conv1D(filters=filters*2, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))
    
    # Layer 5. Number of filters doubles
    model.add(Conv1D(filters=filters*4, kernel_size=kernel_size))
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


def build_cnn_1D_4layer(n_timesteps, n_features,filters=4,kernel_size=5,n_dense=32,dropout=0.5,ytype='Cat'):

    '''
    1D CNN model with 2 CNN layers and 2 dense layers.
    Dropout included for regularisation
    '''

    model = Sequential()

    # Layer 1
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=kernel_size, input_shape=(n_timesteps,n_features)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))

    # Layer 2. 
    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))
    
    # Layer 3. Number of filters doubles
    model.add(Conv1D(filters=filters*2, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))
    
    # Layer 4. Number of filters doubles
    model.add(Conv1D(filters=filters*2, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))
    
    # # Layer 5. Number of filters doubles
    # model.add(Conv1D(filters=filters*4, kernel_size=kernel_size))
    # model.add(BatchNormalization())
    # model.add(ReLU())
    # model.add(Dropout(dropout))
    # model.add(MaxPooling1D(pool_size=2))

    # Layer 3. Flatten and dense
    model.add(Flatten())
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dense(n_dense//2, activation='relu'))
    
    if ytype=='Cat':
        model.add(Dense(1,activation='sigmoid'))
    else:
        model.add(Dense(1,activation='linear',use_bias=False))

    return model


def build_2head_cnn(n_timesteps,n_features,filters=16,kernel_size=128,n_dense=32,dropout=0.5,ytype='Cat'):

    '''
    Multiheaded 1D CNN model with each branch having 2 CNN layers and a flattening layer.
    Dropout included for regularisation
    Flattened layers are merged into dense layers
    '''


    # Head 1 - 1D CNN model. Layer 1
    visible1=Input(shape=(n_timesteps,n_features))
    cnn1=Conv1D(filters=filters, kernel_size=kernel_size)(visible1)
    cnn1=BatchNormalization()(cnn1)
    cnn1=ReLU()(cnn1)
    cnn1=Dropout(dropout)(cnn1)
    cnn1=MaxPooling1D(pool_size=2)(cnn1)
    # Layer 2
    cnn1=Conv1D(filters=filters, kernel_size=kernel_size)(cnn1)
    cnn1=BatchNormalization()(cnn1)
    cnn1=ReLU()(cnn1)
    cnn1=Dropout(dropout)(cnn1)
    cnn1=MaxPooling1D(pool_size=2)(cnn1)
    cnn1=Flatten()(cnn1)
    
    # Head 2 - 1D CNN model. Layer 1
    visible2=Input(shape=(n_timesteps,n_features))
    cnn2=Conv1D(filters=filters, kernel_size=kernel_size)(visible2)
    cnn2=BatchNormalization()(cnn2)
    cnn2=ReLU()(cnn2)
    cnn2=Dropout(dropout)(cnn2)
    cnn2=MaxPooling1D(pool_size=2)(cnn2)
    # Layer 2
    cnn2=Conv1D(filters=filters, kernel_size=kernel_size)(cnn2)
    cnn2=BatchNormalization()(cnn2)
    cnn2=ReLU()(cnn2)
    cnn2=Dropout(dropout)(cnn2)
    cnn2=MaxPooling1D(pool_size=2)(cnn2)
    cnn2=Flatten()(cnn2)

    # Merge and dense layer
    merged=Concatenate(axis=1)([cnn1,cnn2])
    dense_layer = Dense(n_dense,activation='relu')(merged)

    if ytype=='Cat':
        output = Dense(1,activation='sigmoid')(dense_layer)
    else:
        output = Dense(1,activation='linear')(dense_layer)

    model = Model(inputs=[visible1,visible2],outputs=output)

    return model