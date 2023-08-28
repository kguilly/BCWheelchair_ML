'''

This file was developed by Kaleb Guillot for the purposes of the 
senior design project: The Brain-Controlled Wheelchair

to run this file from the command line: 
(env) Documents/GitHub/BCWheelchair_ML $ python -m tests/test_1.py

'''
import numpy as np

# EEGNet specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K


################################################################
## Load the dataset and order it 



################################################################
## Process, filter, and epoch the data



################################################################
## Call EEGNet