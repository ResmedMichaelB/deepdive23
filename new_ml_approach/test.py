import numpy as np
import json, os, sys, glob
import scipy as sp
import matplotlib.pyplot as plt


cur_path=os.getcwd()
sys.path.append(cur_path)

from helpers.load_data import *
from helpers.plotting import plot_oct

if __name__ == '__main__':

    folder_path = r'C:\Users\MichaelB9\Data\S9 AHI\unfiltered'
    # folder_path = r'A:\dynamic\datasets\Ouler\Autoset CPAP Filius study at SCSDC\Input_data_position'
    # folder_path = r'C:\Users\MichaelB9\Data\Fillius\filtered'

    target_signal='Position'
    input_signals=['Flow']#,'Mask Pres']

    json_files = glob.glob(os.path.join(folder_path,'*.json'))
    # json_files = [js for js in json_files if 'ecg' not in js]

    sig_tensor, target_tensor, subject_names = load_sig_tensor(
        json_files[0:10],epoch_size = 60,epoch_step = 30,
        sample_rate = 25, input_signals=input_signals,
        target_signal=target_signal,
        dataset='s9',
        )

    split(sig_tensor['flow'], target_tensor, subject_names, train_size=0.8, val_size=0.1,test_size=0.1)

    print('here')

