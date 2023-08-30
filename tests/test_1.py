'''

This file was developed by Kaleb Guillot for the purposes of the 
senior design project: The Brain-Controlled Wheelchair

to run this file from the command line: 
(env) Documents/GitHub/BCWheelchair_ML $ python -m tests/test_1.py


Questions that I have to answer: 
- How long of durations do I want to sample? 
- How do I want to separate training and testing? 
'''
import numpy as np
import os
import pyedflib

# EEGNet specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K


kernels, channels, samples = 1, 64, 1

################################################################
## Load the dataset and order it 
data_path = "/home/kaleb/Documents/eeg_dataset/files/S001/" # grabbing one subject
all_files = os.listdir(data_path)

edf_files = [file for file in all_files if file.endswith('.edf')] # grab each of the files ending in '.edf'

all_X = []
all_y = []

for file in edf_files: # grab all the relevant data from the edf files
    path = os.path.join(data_path, file)
    edf_data = pyedflib.EdfReader(path)
    
    X = []
    for channel in range(channels):
        arr = edf_data.readSignal(channel)
        X.append(arr)
        
    edf_data.close()

    # after reading from a file, get the label array
    label = int(path[-6] + path[-5])
    y = np.full_like(X, label)

    all_X.append(X)
    all_y.append(y)
    

print('done')


################################################################
## Process, filter, and epoch the data



################################################################
## Call EEGNet