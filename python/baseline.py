import getpass
import sys
import time

import numpy as np
from copy import deepcopy
import random

import tensorflow as tf
from get_babi_data import get_task_1_train
from get_babi_data import get_task_1_test
from tensorflow.python.ops.seq2seq import sequence_loss

#### MODEL PARAMETERS ####


#### END MODEL PARAMETERS ####

def split_training_data(train_total):

    # Set seed for consistent splitting
    random.seed(31415)
    np.random.seed(9265)

    np.random.shuffle(train_total)
    split_index = int(len(train_total)*0.8)

    train = train_total[:split_index]
    dev = train_total[split_index:]

    return train, dev

def run_baseline():

    # Get train dataset for task 1
    train_total = get_task_1_train()

    train, validation = split_training_data(train_total)

    # Get test dataset for task 1
    test = get_task_1_test()

    # Print summary statistics
    print "Training samples: {}".format(len(train))
    print "Validation samples: {}".format(len(validation))
    print "Testing samples: {}".format(len(test))

if __name__ == "__main__":
    run_baseline()
