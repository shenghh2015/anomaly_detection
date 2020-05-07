import os
import numpy as np
from sklearn.metrics import roc_auc_score

from load_data import *

dataset_list = ['dense', 'hetero', 'scattered', 'fatty', 'total']

# print out a summary of the datasets
# evaluate_target_HO(dataset = 'dense', train = 7400, valid = 100, test = 400)
# evaluate_target_HO(dataset = 'hetero', train = 36000, valid = 100, test = 400)
# evaluate_target_HO(dataset = 'scattered', train = 33000, valid = 100, test = 400)
# evaluate_target_HO(dataset = 'fatty', train = 9000, valid = 100, test = 400)
# evaluate_target_HO(dataset = 'total', train = 80000, valid = 100, test = 400)
evaluate_target_HO(dataset = 'total', train = 15000, valid = 100, test = 400)
