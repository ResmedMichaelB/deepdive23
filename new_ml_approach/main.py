import numpy as np
import json, os, sys, glob
import scipy as sp
import matplotlib.pyplot as plt


cur_path=os.getcwd()
sys.path.append(cur_path)

from load_data import *
from plotting import plot_oct

if __name__ == '__main__':

    # folder_path = r'C:\Users\MichaelB9\Documents\POSA data\S9 AHI\json_files\preprocessed'
    folder_path = r'A:\dynamic\datasets\Ouler\Autoset CPAP Filius study at SCSDC\Input_data_position'
    folder_path = r'C:\Users\MichaelB9\Documents\POSA data\CPAP\Fillius\unprocessed'

    target_signal='heartrate'
    input_signals=['flow']#,'Mask Pres']

    json_files = glob.glob(os.path.join(folder_path,'*.json'))
    # json_files = [js for js in json_files if 'ecg' not in js]

    sig_tensor, target_tensor, subject_names = load_sig_tensor(
        json_files,epoch_size = 30,epoch_step = 30,
        sample_rate = 25, input_signals=input_signals,
        target_signal=target_signal,
        dataset='fillius',
        ytype='Num'
        )



    print('here')

