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

TRAINING_SPLIT = 0.8
WORD_VECTOR_LENGTH = 50
VOCAB_LENGTH = 10000

#### END MODEL PARAMETERS ####

def split_training_data(train_total):

    # Set seed for consistent splitting
    random.seed(31415)
    np.random.seed(9265)

    np.random.shuffle(train_total)
    split_index = int(len(train_total)*TRAINING_SPLIT)

    train = train_total[:split_index]
    dev = train_total[split_index:]

    return train, dev

def add_placeholders():
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables

      input_placeholder
      question_placeholder
      labels_placeholder

    """

    # TODO figure out what shapes these should be exactly
    input_placeholder = tf.placeholder(tf.int32, shape=[None, WORD_VECTOR_LENGTH])
    question_placeholder = tf.placeholder(tf.int32, shape=[None, WORD_VECTOR_LENGTH])
    labels_placeholder = tf.placeholder(tf.int32, shape=[None, VOCAB_LENGTH])
    return input_placeholder, question_placeholder, labels_placeholder

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

    # Get glove vectors

    # Add placeholders
    input_placeholder, question_placeholder, labels_placeholder = add_placeholders()

    # Initialize input model

    # Initialize question model

    # Initialize answer model

    # Compute loss

    # Add optimizer

    # Create feed dicts with inputs

    # Train over multiple epochs

    # Test

if __name__ == "__main__":
    run_baseline()
