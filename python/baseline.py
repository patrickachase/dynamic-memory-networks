import getpass
import sys
import time

import numpy as np
from copy import deepcopy
import random

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from get_babi_data import get_task_6_train
from get_babi_data import get_task_6_test
from get_glove import load_glove_vectors
from get_babi_data import get_task_1_train
from get_babi_data import get_task_1_test
from tensorflow.python.ops.seq2seq import sequence_loss

#### MODEL PARAMETERS ####

TRAINING_SPLIT = 0.8
WORD_VECTOR_LENGTH = 50
VOCAB_LENGTH = 10000
LEARNING_RATE = 0.001
NUM_CLASSES = 2
NUM_EPOCHS = 2
HIDDEN_SIZE = 50

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
    labels_placeholder = tf.placeholder(tf.int32, shape=[None, NUM_CLASSES])
    return input_placeholder, question_placeholder, labels_placeholder

def format_data(data, glove_dict):
    text_arr = []
    question_arr = []
    answer_arr = []
    for (text, question, answer) in data:
        # convert word array to word vector array for text
        text_vec = [] 
        for word in text:
            if word in glove_dict:
                wordvec = glove_dict[word]
            else:
                wordvec = np.random.rand(1, WORD_VECTOR_LENGTH)[0]
                wordvec /= np.sum(wordvec)
            text_vec.append(wordvec)

        question_arr.append(text_vec)

        # convert word array to word vector array for question
        question_vec = [] 
        for word in question:
            if word in glove_dict:
                wordvec = glove_dict[word]
            else:
                wordvec = np.random.rand(1, WORD_VECTOR_LENGTH)[0]
                wordvec /= np.sum(wordvec)
            question_vec.append(wordvec)

        text_arr.append(question_vec)

        # convert answer to a onehot vector
        if answer == 'yes':
            answer_arr.append([1, 0])
        else:
            answer_arr.append([0, 1])

    return text_arr, question_arr, answer_arr

def RNN(X, initial_state, W_hidden, b_hidden, W_out, b_out):
  # TODO conver tensor (matrix) X to a list of tensors
  state = initial_state
  lstm_cell = rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

  print X
  for i in xrange(len(X)):
    x_scaled = tf.matmul(X[i], W_hidden) + b_hidden
    output, state = lstm_cell(lstm_cell, x_scaled, initial_state=state)

  return tf.matmul(output, W_out) + b_out

def run_baseline():

    # Get train dataset for task 1
    train_total = get_task_6_train()

    train, validation = split_training_data(train_total)

    # Get test dataset for task 1
    test = get_task_6_test()

    # Print summary statistics
    print "Training samples: {}".format(len(train))
    print "Validation samples: {}".format(len(validation))
    print "Testing samples: {}".format(len(test))

    # Get word to glove vectors dictionary
    glove_dict = load_glove_vectors()

    # Get data into word vector format
    text_train, question_train, answer_train = format_data(train, glove_dict)
    text_val, question_val, answer_val = format_data(validation, glove_dict)
    text_test, question_test, answer_test = format_data(test, glove_dict)

    # Add placeholders
    input_placeholder, question_placeholder, labels_placeholder = add_placeholders()

    # Initialize input model
    with tf.variable_scope("text"):
        W_hidden = tf.get_variable("W_hidden", shape=(WORD_VECTOR_LENGTH, HIDDEN_SIZE))
        b_hidden = tf.get_variable("b_hidden", shape=(HIDDEN_SIZE, 1))
        W_out = tf.get_variable("W_out", shape=(HIDDEN_SIZE, NUM_CLASSES))
        b_out = tf.get_variable("b_out", shape=(NUM_CLASSES, 1))

    initial_state = np.zeros(HIDDEN_SIZE)
    pred = RNN(input_placeholder, initial_state, W_hidden, b_hidden, W_out, b_out)

    # Initialize question model
    # Initialize answer model

    # Compute loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    # Add optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Create feed dicts with inputs
    init = tf.initialize_all_variables()

    # Train over multiple epochs
    with tf.Session() as sess:
      sess.run(init)
      # train until we reach the maximum number of epochs
      for epoch in range(MAX_EPOCHS):
        for i in range(len(train)):
            sess.run(optimizer, feed_dict={input_placeholder: text_train[i], labels_placeholder: answer_train[i], initial_state: np.zeros(HIDDEN_SIZE)})

    # Test
    sess.run(accuracy, feed_dict={input_placeholder: text_val, labels_placeholder: answer_val, initial_state: np.zeros(HIDDEN_SIZE)})

if __name__ == "__main__":
    run_baseline()
