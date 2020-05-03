import tensorflow as tf
import tensorflow.math as tm

import numpy as np
import os
import glob
from natsort import natsorted
from termcolor import colored 
import argparse
from sklearn.metrics import roc_auc_score
import scipy.io

from load_data import *
from model import *

## load dataset
X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst = load_anomaly_data(dataset = 'dense', train = 7100, valid = 400, test = 400)
