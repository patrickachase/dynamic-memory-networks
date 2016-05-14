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
HIDDEN_SIZE = 25
EARLY_STOPPING = 2
MAX_INPUT_LENGTH = 40
MAX_EPOCHS = 10

#### END MODEL PARAMETERS ####

def split_training_data(train_total):
  # Set seed for consistent splitting
  random.seed(31415)
  np.random.seed(9265)

  np.random.shuffle(train_total)
  split_index = int(len(train_total) * TRAINING_SPLIT)

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
  input_placeholder = tf.placeholder(tf.float32, shape=[None, WORD_VECTOR_LENGTH])
  question_placeholder = tf.placeholder(tf.float32, shape=[None, WORD_VECTOR_LENGTH])
  labels_placeholder = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
  return input_placeholder, question_placeholder, labels_placeholder


# Takes in the data set with (input, question, answer) tuplets and the dictionary of glove
# vectors and returns the word vectors for the input, question, and answer.
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
      answer = np.array([1, 0])
      answer = answer.reshape((1, NUM_CLASSES))
      answer_arr.append(answer)
    else:
      answer = np.array([0, 1])
      answer = answer.reshape((1, NUM_CLASSES))
      answer_arr.append(answer)

  return text_arr, question_arr, answer_arr


def RNN(X, initial_state, W_hidden, b_hidden, W_out, b_out, num_words_in_X):
  # Reshape `X` as a vector. -1 means "set this dimension automatically".
  X_as_vector = tf.reshape(X, [-1])

  # Create another vector containing zeroes to pad `X` to (MAX_INPUT_LENGTH * WORD_VECTOR_LENGTH) elements.
  zero_padding = tf.zeros([MAX_INPUT_LENGTH * WORD_VECTOR_LENGTH] - tf.shape(X_as_vector), dtype=X.dtype)

  # Concatenate `X_as_vector` with the padding.
  X_padded_as_vector = tf.concat(0, [X_as_vector, zero_padding])

  # Reshape the padded vector to the desired shape.
  X_padded = tf.reshape(X_padded_as_vector, [MAX_INPUT_LENGTH, WORD_VECTOR_LENGTH])

  # Split X into a list of tensors of length MAX_INPUT_LENGTH where each tensor is a 1xWORD_VECTOR_LENGTH vector
  # of the word vectors
  # TODO change input to be a list of tensors of length MAX_INPUT_LENGTH where each tensor is a BATCH_SIZExWORD_VECTOR_LENGTH vector
  X = tf.split(0, MAX_INPUT_LENGTH, X_padded)

  lstm_cell = rnn_cell.LSTMCell(num_units=HIDDEN_SIZE, input_size=WORD_VECTOR_LENGTH)

  # Compute final state after feeding in word vectors
  state = tf.zeros([1, HIDDEN_SIZE])

  # Print out all input tensors
  print "Tensors: \n\n"
  print X
  print state
  print num_words_in_X
  # TODO add termination at num_steps back in with sequence_length parameter
  output, state = rnn.rnn(lstm_cell, X, initial_state=state, dtype=tf.float32)

  return output[-1]


def run_baseline():
  # Get train dataset for task 6
  train_total = get_task_6_train()

  train, validation = split_training_data(train_total)

  # Get test dataset for task 6
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

  input_length = tf.shape(input_placeholder)[0]

  # Initialize input model
  with tf.variable_scope("text"):
    W_hidden = tf.get_variable("W_hidden", shape=(WORD_VECTOR_LENGTH, HIDDEN_SIZE))
    b_hidden = tf.get_variable("b_hidden", shape=(1, HIDDEN_SIZE))
    W_out = tf.get_variable("W_out", shape=(HIDDEN_SIZE, NUM_CLASSES))
    b_out = tf.get_variable("b_out", shape=(1, NUM_CLASSES))

  # TODO should the initial state be a placeholder?
  initial_state = np.zeros([1,HIDDEN_SIZE])

  final_state = RNN(input_placeholder, initial_state, W_hidden, b_hidden, W_out, b_out, input_length)

  print "Final state: \n\n"
  print final_state

  print W_out
  print b_out
  pred = tf.nn.softmax(tf.matmul(final_state, W_out) + b_out)

  # Initialize question model

  # Initialize answer model

  # Compute loss
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, labels_placeholder))

  # Add optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

  # Initialize all variables
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  # Train over multiple epochs
  with tf.Session() as sess:
    best_loss = float('inf')
    best_val_epoch = 0

    sess.run(init)
    # train until we reach the maximum number of epochs
    for epoch in range(MAX_EPOCHS):

      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###

      total_training_loss = 0
      # Compute average loss on training data
      for i in range(len(train)):
        print answer_train[i]
        print np.shape(answer_train[i])
        cost = sess.run(optimizer, feed_dict={input_placeholder: text_train[i], labels_placeholder: answer_train[i]})
        training_loss = training_loss + cost

      average_training_loss = total_training_loss / len(train)

      # TODO Compute average loss on validation set
      validation_loss = float('inf')

      print 'Training loss: {}'.format(average_training_loss)
      print 'Validation loss: {}'.format(validation_loss)
      if validation_loss < best_loss:
        best_loss = validation_loss
        best_val_epoch = epoch
        saver.save(sess, './weights/rnn.weights')
      if epoch - best_val_epoch > EARLY_STOPPING:
        break
      print 'Total time: {}'.format(time.time() - start)

  # Compute average loss on testing data with best weights
  saver.restore(sess, './weights/rnn.weights')

  sess.run(accuracy,
           feed_dict={input_placeholder: text_val, labels_placeholder: answer_val,
                      initial_state: np.zeros(HIDDEN_SIZE)})

  print '=-=' * 5
  print 'Test perplexity: {}'.format(accuracy)
  print '=-=' * 5

  # TODO add input loop so we can test and debug with our own examples
  input = ""
  while input:
    # Run model

    input = raw_input('> ')


if __name__ == "__main__":
  run_baseline()
