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
from format_data import split_training_data
from format_data import format_data

#### MODEL PARAMETERS ####

TRAINING_SPLIT = 0.8
WORD_VECTOR_LENGTH = 50
VOCAB_LENGTH = 10000
LEARNING_RATE = 0.001
NUM_CLASSES = 2
NUM_EPOCHS = 2
INPUT_HIDDEN_SIZE = 25
QUESTION_HIDDEN_SIZE = 25
EARLY_STOPPING = 2
MAX_INPUT_LENGTH = 200
MAX_QUESTION_LENGTH = 100
MAX_EPOCHS = 10
BATCH_SIZE = 1

# Number of training elements to train on before an update is printed
UPDATE_LENGTH = 100


#### END MODEL PARAMETERS ####

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
  input_length_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
  question_placeholder = tf.placeholder(tf.float32, shape=[None, WORD_VECTOR_LENGTH])
  question_length_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
  labels_placeholder = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
  return input_placeholder, input_length_placeholder, question_placeholder, question_length_placeholder, labels_placeholder


def RNN(X, num_words_in_X, hidden_size, max_input_size):
  # Reshape `X` as a vector. -1 means "set this dimension automatically".
  X_as_vector = tf.reshape(X, [-1])

  # Create another vector containing zeroes to pad `X` to (MAX_INPUT_LENGTH * WORD_VECTOR_LENGTH) elements.
  zero_padding = tf.zeros([max_input_size * WORD_VECTOR_LENGTH] - tf.shape(X_as_vector), dtype=X.dtype)

  # Concatenate `X_as_vector` with the padding.
  X_padded_as_vector = tf.concat(0, [X_as_vector, zero_padding])

  # Reshape the padded vector to the desired shape.
  X_padded = tf.reshape(X_padded_as_vector, [max_input_size, WORD_VECTOR_LENGTH])

  # Split X into a list of tensors of length MAX_INPUT_LENGTH where each tensor is a 1xWORD_VECTOR_LENGTH vector
  # of the word vectors
  # TODO change input to be a list of tensors of length MAX_INPUT_LENGTH where each tensor is a BATCH_SIZExWORD_VECTOR_LENGTH vector
  X = tf.split(0, max_input_size, X_padded)

  lstm_cell = rnn_cell.LSTMCell(num_units=hidden_size, input_size=WORD_VECTOR_LENGTH)

  # Print out all input tensors
  # print "Tensors: \n\n"
  # print X
  # print num_words_in_X
  # TODO add termination at num_steps back in with sequence_length parameter
  # TODO add back initial state
  output, state = rnn.rnn(lstm_cell, X, sequence_length=num_words_in_X, dtype=tf.float32)

  # TODO figure out if we should return state
  return output[-1]


def count_positive_and_negative(answer_vecs):
  num_positive = 0
  for answer_vec in answer_vecs:

    if answer_vec[0, 0] == 1:
      num_positive = num_positive + 1

  num_negative = len(answer_vecs) - num_positive

  return num_positive, num_negative


def run_baseline():
  # Get train dataset for task 6
  train_total = get_task_6_train()

  train, validation = split_training_data(train_total)

  # Get test dataset for task 6
  test = get_task_6_test()

  # Get word to glove vectors dictionary
  glove_dict = load_glove_vectors()

  # Get data into word vector format
  text_train, question_train, answer_train = format_data(train, glove_dict)
  text_val, question_val, answer_val = format_data(validation, glove_dict)
  text_test, question_test, answer_test = format_data(test, glove_dict)

  num_positive_train, num_negative_train = count_positive_and_negative(answer_train)

  # Print summary statistics
  print "Training samples: {}".format(len(train))
  print "Positive training samples: {}".format(num_positive_train)
  print "Negative training samples: {}".format(num_negative_train)
  print "Validation samples: {}".format(len(validation))
  print "Testing samples: {}".format(len(test))

  # Add placeholders
  input_placeholder, input_length_placeholder, question_placeholder, question_length_placeholder, labels_placeholder, = add_placeholders()

  # Initialize answer model
  with tf.variable_scope("output"):
    W_out = tf.get_variable("W_out", shape=(INPUT_HIDDEN_SIZE + QUESTION_HIDDEN_SIZE, NUM_CLASSES))
    b_out = tf.get_variable("b_out", shape=(1, NUM_CLASSES))

  # Initialize question model

  with tf.variable_scope("input"):
    input_state = RNN(input_placeholder, input_length_placeholder, INPUT_HIDDEN_SIZE, MAX_INPUT_LENGTH)

  with tf.variable_scope("question"):
    question_state = RNN(question_placeholder, question_length_placeholder, QUESTION_HIDDEN_SIZE, MAX_QUESTION_LENGTH)

  # Concatenate input and question vectors
  input_and_question = tf.concat(1, [input_state, question_state])

  # Answer model
  prediction_probs = tf.nn.softmax(tf.matmul(input_and_question, W_out) + b_out)

  # To get predictions perform a max over each row
  prediction = tf.argmax(prediction_probs, 1)

  # Compute loss
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction_probs, labels_placeholder))

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
      num_correct = 0

      prev_prediction = 0

      # Compute average loss on training data
      for i in range(len(train)):

        num_words_in_inputs = [np.shape(text_train[i])[0]]
        num_words_in_question = [np.shape(question_train[i])[0]]
        loss, current_pred, probs, _ = sess.run([cost, prediction, prediction_probs, optimizer],
                                                feed_dict={input_placeholder: text_train[i],
                                                           input_length_placeholder: num_words_in_inputs,
                                                           question_placeholder: question_train[i],
                                                           question_length_placeholder: num_words_in_question,
                                                           labels_placeholder: answer_train[i]})

        # print "Current pred probs: {}".format(probs)
        # print "Current pred: {}".format(currentPred[0])
        # print "Current answer: {}".format(np.argmax(answer_train[i]))

        if current_pred[0] == np.argmax(answer_train[i]):
          num_correct = num_correct + 1

        # Print a training update
        if i % UPDATE_LENGTH == 0:
          print "Current average training loss: {}".format(total_training_loss / (i + 1))
          print "Current training accuracy: {}".format(float(num_correct) / (i + 1))

        total_training_loss = total_training_loss + loss

        # Check if prediction changed
        if prev_prediction != current_pred[0]:
          print "Prediction changed"

        prev_prediction = current_pred[0]

      average_training_loss = total_training_loss / len(train)
      training_accuracy = float(num_correct) / len(train)

      validation_loss = float('inf')

      total_validation_loss = 0
      num_correct_val = 0
      # Compute average loss on validation data
      for i in range(len(validation)):
        num_words_in_inputs = [np.shape(answer_train[i])[0]]
        loss, currentPred, probs, _ = sess.run([cost, prediction, prediction_probs, optimizer],
                                               feed_dict={input_placeholder: text_val[i],
                                                          question_placeholder: question_val[i],
                                                          labels_placeholder: answer_val[i],
                                                          input_length_placeholder: num_words_in_inputs})

        if currentPred == np.argmax(answer_val[i]):
          num_correct_val = num_correct_val + 1

        total_validation_loss = total_validation_loss + loss

      average_validation_loss = total_validation_loss / len(validation)
      validation_accuracy = float(num_correct_val) / len(validation)

      print 'Training loss: {}'.format(average_training_loss)
      print 'Training accuracy: {}'.format(training_accuracy)
      print 'Validation loss: {}'.format(average_validation_loss)
      print 'Validation accuracy: {}'.format(validation_accuracy)
      if validation_loss < best_loss:
        best_loss = validation_loss
        best_val_epoch = epoch
        saver.save(sess, '../data/weights/rnn.weights')
      if epoch - best_val_epoch > EARLY_STOPPING:
        break
      print 'Total time: {}'.format(time.time() - start)

  # Compute average loss on testing data with best weights
  saver.restore(sess, '../data/weights/rnn.weights')

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
