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
from random import shuffle

#### MODEL PARAMETERS ####

WORD_VECTOR_LENGTH = 50
VOCAB_LENGTH = 10000
LEARNING_RATE = 0.001
NUM_CLASSES = 2
INPUT_HIDDEN_SIZE = 50
QUESTION_HIDDEN_SIZE = 50
ANSWER_HIDDEN_SIZE = 50
MAX_INPUT_LENGTH = 100
MAX_QUESTION_LENGTH = 20
MAX_EPOCHS = 100

BATCH_SIZE = 100

# Number of batches to train on before an update is printed
UPDATE_LENGTH = 1


#### END MODEL PARAMETERS ####

def add_placeholders():
  """Generate placeholder variables to represent the input tensors

  These placeholders are used as inputs by the rest of the model building
  code and will be fed data during training.  Note that when "None" is in a
  placeholder's shape, it's flexible

  Adds following nodes to the computational graph.
  (When None is in a placeholder's shape, it's flexible)

  input_placeholder: Input is (max number of words) x (BATCH_SIZE) x (WORD_VECTOR_LENGTH)
  input_length_placeholder: Number of words in each example in the batch
  question_placeholder: Question is (max number of words) x (BATCH_SIZE) x (WORD_VECTOR_LENGTH)
  question_length_placeholder: Number of words in each example in the batch
  labels_placeholder: Answers for each example (BATCH_SIZE) x (NUM_CLASSES)
  dropout_placeholder: Dropout value placeholder (scalar),
                       type tf.float32

  """

  input_placeholder = tf.placeholder(tf.float32, shape=[MAX_INPUT_LENGTH, BATCH_SIZE, WORD_VECTOR_LENGTH])
  input_length_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
  question_placeholder = tf.placeholder(tf.float32, shape=[MAX_QUESTION_LENGTH, BATCH_SIZE, WORD_VECTOR_LENGTH])
  question_length_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
  labels_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
  return input_placeholder, input_length_placeholder, question_placeholder, question_length_placeholder, labels_placeholder


def RNN(X, num_words_in_X, hidden_size, input_vector_size, max_input_size):
  """
  Passes the input data through an RNN and outputs the final states.

  X: Input is a MAX_INPUT_LENGTH X BATCH_SIZE X WORD_VECTOR_LENGTH matrix
  num_words_in_X: Number of words in X, which is needed because X is zero padded
  hidden_size: The dimensionality of the hidden layer of the RNN
  input_vector_size: This is the dimensionality of each input vector, in this case it is WORD_VECTOR_LENGTH 
  max_input_size: This is the max number of input vectors that can be passed in to the RNN.

  """

  # Split X into a list of tensors of length max_input_size where each tensor is a BATCH_SIZE x input_vector_size vector
  X = tf.split(0, max_input_size, X)

  squeezed = []
  for i in range(len(X)):
    squeezed.append(tf.squeeze(X[i]))

  gru_cell = rnn_cell.GRUCell(num_units=hidden_size, input_size=input_vector_size)
  output, state = rnn.rnn(gru_cell, squeezed, sequence_length=num_words_in_X, dtype=tf.float32)
  return output, state, X


def batch_data(data):
  """ 
  Takes in a list of tuples of (input, question, answer) and returns a list of length 
  (number of examples) / BATCH_SIZE where each element is a batch of BATCH_SIZE questions.

  data: a list of tuples of (input, question, answer)

  """

  # Compute total number of batches for the data set
  num_batches = len(data) / BATCH_SIZE

  batched_data = []
  for i in range(num_batches):
    # Get current batch
    current_batch = []

    for j in range(BATCH_SIZE):
      if i * BATCH_SIZE + j < len(data):
        current_batch.append(data[i * BATCH_SIZE + j])

    batched_data.append(current_batch)

  return batched_data


def convert_to_vectors(batched_data, glove_dict):
  """

  Takes in a list of batches of data and converts them to a list of batched vectors
  Each element of the returned list contains all the vectors for a batch of data
  Output dimension is (max num words) x (BATCH_SIZE) x (WORD_VECTOR_LENGTH)
  If there are no fewer words than the max number of words zero vectors are added for padding

  batched_data: A list of of length number of batches. Each element is a batch of 
                (input, question, answer) tuples
  glove_dict: dictionary from word to glove word vector

  """


  batched_input_vecs = []
  batched_input_lengths = []
  batched_question_vecs = []
  batched_question_lengths = []
  batched_answer_vecs = []

  for batch in batched_data:

    # Batch is a list of tuples of length BATCH_SIZE or less

    # Create an array to hold all of the word vectors for the batch
    batch_input_vecs = np.zeros((MAX_INPUT_LENGTH, len(batch), WORD_VECTOR_LENGTH))
    batch_input_lengths = np.zeros(len(batch))
    batch_question_vecs = np.zeros((MAX_QUESTION_LENGTH, len(batch), WORD_VECTOR_LENGTH))
    batch_question_lengths = np.zeros(len(batch))
    batch_answer_vecs = np.zeros((len(batch), NUM_CLASSES))

    for i in range(len(batch)):
      example = batch[i]
      input = example[0]
      question = example[1]
      answer = example[2]

      # Add input vectors
      for j in range(len(input)):
        if j >= MAX_INPUT_LENGTH:
          continue
        word = input[j]

        word_vector = get_word_vector(word, glove_dict)

        # Set the jth word of the ith batch to be the word vector
        batch_input_vecs[j, i, :] = word_vector

      # Add input length
      batch_input_lengths[i] = len(input)

      # Add question vectors
      for j in range(len(question)):
        if j >= MAX_QUESTION_LENGTH:
          continue
        word = question[j]

        word_vector = get_word_vector(word, glove_dict)

        # Set the jth word of the ith batch to be the word vector
        batch_question_vecs[j, i, :] = word_vector

      # Add question length
      batch_question_lengths[i] = len(question)

      # Add answer vectors

      # convert answer to a onehot vector
      if answer == 'yes':
        answer = np.array([1, 0])
        answer = answer.reshape((1, NUM_CLASSES))
      else:
        answer = np.array([0, 1])
        answer = answer.reshape((1, NUM_CLASSES))

      batch_answer_vecs[i, :] = answer

    batched_input_vecs.append(batch_input_vecs)
    batched_input_lengths.append(batch_input_lengths)
    batched_question_vecs.append(batch_question_vecs)
    batched_question_lengths.append(batch_question_lengths)
    batched_answer_vecs.append(batch_answer_vecs)

  return batched_input_vecs, batched_input_lengths, batched_question_vecs, batched_question_lengths, batched_answer_vecs


def get_word_vector(word, glove_dict):
  """ 
  Helper function that returns a glove vector for a word if it exists in the glove dictionary, and
  returns a random vector if it does not. 

  word: The string of a word to look up in the dictionary
  glove_dict: A dictionary from words to glove word vectors

  """

  if word in glove_dict:
    word_vec = glove_dict[word]
  else:
    word_vec = np.random.rand(1, WORD_VECTOR_LENGTH)[0]
    word_vec /= np.sum(word_vec)
  return word_vec


def answer_module(input_and_question):
  """ 
  The answer module is a NN with the following structure:
  1. dense fully connected layer
  2. ReLU activation
  3. dense fully connected layer
  Returns the projections which is an array of size NUM_CLASSES

  input_and_quesiton: The concatenated outputs from the input and question modules

  """

  with tf.variable_scope("answer_module"):
    W_1 = tf.get_variable("W_1", shape=(INPUT_HIDDEN_SIZE + QUESTION_HIDDEN_SIZE, ANSWER_HIDDEN_SIZE))
    b_1 = tf.get_variable("b_1", shape=(1, ANSWER_HIDDEN_SIZE))

    W_out = tf.get_variable("W_out", shape=(ANSWER_HIDDEN_SIZE, NUM_CLASSES))
    b_out = tf.get_variable("b_out", shape=(1, NUM_CLASSES))

  h = tf.nn.relu(tf.matmul(input_and_question, W_1) + b_1)

  projections = tf.matmul(h, W_out) + b_out

  return projections


def run_baseline():
  """
  Main function which loads in data, runs the model, and prints out statistics

  """
  # Get train dataset for task 6
  train_total = get_task_6_train()

  train, validation = split_training_data(train_total)

  # Get test dataset for task 6
  test = get_task_6_test()

  # Get word to glove vectors dictionary
  glove_dict = load_glove_vectors()

  # Split data into batches
  train_batches = batch_data(train)
  validation_batches = batch_data(validation)
  test_batches = batch_data(test)

  # Convert batches into vectors
  train_batched_input_vecs, train_batched_input_lengths, train_batched_question_vecs, \
  train_batched_question_lengths, train_batched_answer_vecs = convert_to_vectors(
    train_batches, glove_dict)

  val_batched_input_vecs, val_batched_input_lengths, val_batched_question_vecs, \
  val_batched_question_lengths, val_batched_answer_vecs = convert_to_vectors(validation_batches, glove_dict)

  test_batched_input_vecs, test_batched_input_lengths, test_batched_question_vecs, \
  test_batched_question_lengths, test_batched_answer_vecs = convert_to_vectors(
    test_batches, glove_dict)

  # Print summary statistics
  print "Training samples: {}".format(len(train))
  print "Validation samples: {}".format(len(validation))
  print "Testing samples: {}".format(len(test))
  print "Batch size: {}".format(BATCH_SIZE)
  print "Training number of batches: {}".format(len(train_batches))
  print "Validation number of batches: {}".format(len(validation_batches))
  print "Test number of batches: {}".format(len(test_batches))

  # Add placeholders
  input_placeholder, input_length_placeholder, question_placeholder, question_length_placeholder, labels_placeholder = add_placeholders()

  # Initialize input module
  with tf.variable_scope("input"):
    input_output, input_state, X_input = RNN(input_placeholder, input_length_placeholder, INPUT_HIDDEN_SIZE,
                                             WORD_VECTOR_LENGTH,
                                             MAX_INPUT_LENGTH)
  # Initialize question module
  with tf.variable_scope("question"):
    question_output, question_state, Q_input = RNN(question_placeholder, question_length_placeholder,
                                                   QUESTION_HIDDEN_SIZE, WORD_VECTOR_LENGTH, MAX_QUESTION_LENGTH)

  # Concatenate input and question vectors
  input_and_question = tf.concat(1, [input_state, question_state])

  # Answer model
  with tf.variable_scope("answer"):
    projections = answer_module(input_and_question)

  prediction_probs = tf.nn.softmax(projections)

  # Compute loss
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(projections, labels_placeholder))

  # Add optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

  # Initialize all variables
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  # Train over multiple epochs
  with tf.Session() as sess:
    best_validation_accuracy = float('inf')
    best_val_epoch = 0

    sess.run(init)
    # train until we reach the maximum number of epochs
    for epoch in range(MAX_EPOCHS):

      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###

      total_training_loss = 0
      sum_training_accuracy = 0

      # Compute average loss on training data
      for i in range(len(train_batches)):

        loss, _, batch_prediction_probs = sess.run(
          [cost, optimizer, prediction_probs],
          feed_dict={input_placeholder: train_batched_input_vecs[i],
                     input_length_placeholder: train_batched_input_lengths[i],
                     question_placeholder: train_batched_question_vecs[i],
                     question_length_placeholder: train_batched_question_lengths[i],
                     labels_placeholder: train_batched_answer_vecs[i]})

        total_training_loss += loss

        batch_accuracy = np.equal(np.argmax(batch_prediction_probs, axis=1),
                                  np.argmax(train_batched_answer_vecs[i], axis=1)).mean()

        sum_training_accuracy += batch_accuracy

        # Print a training update
        if i % UPDATE_LENGTH == 0:
          print "Current average training loss: {}".format(total_training_loss / (i + 1))
          print "Current training accuracy: {}".format(sum_training_accuracy / (i + 1))

      average_training_loss = total_training_loss / len(train_batches)
      training_accuracy = sum_training_accuracy / len(train_batches)

      total_validation_loss = 0
      sum_validation_accuracy = 0

      # Compute average loss on validation data
      for i in range(len(validation_batches)):
        loss, batch_prediction_probs = sess.run(
          [cost, prediction_probs],
          feed_dict={input_placeholder: val_batched_input_vecs[i],
                     input_length_placeholder: val_batched_input_lengths[i],
                     question_placeholder: val_batched_question_vecs[i],
                     question_length_placeholder: val_batched_question_lengths[i],
                     labels_placeholder: val_batched_answer_vecs[i]})

        total_validation_loss = total_validation_loss + loss

        batch_accuracy = np.equal(np.argmax(batch_prediction_probs, axis=1),
                                  np.argmax(val_batched_answer_vecs[i], axis=1)).mean()

        sum_validation_accuracy += batch_accuracy

      average_validation_loss = total_validation_loss / len(validation_batches)
      validation_accuracy = sum_validation_accuracy / len(validation_batches)

      print 'Training loss: {}'.format(average_training_loss)
      print 'Training accuracy: {}'.format(training_accuracy)
      print 'Validation loss: {}'.format(average_validation_loss)
      print 'Validation accuracy: {}'.format(validation_accuracy)
      if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_val_epoch = epoch
        saver.save(sess, '../data/weights/rnn.weights')
        print "Weights saved"

      print 'Total time: {}'.format(time.time() - start)

    # Compute average loss on testing data with best weights
    saver.restore(sess, '../data/weights/rnn.weights')

    total_test_loss = 0
    sum_test_accuracy = 0

    # Compute average loss on test data
    for i in range(len(test_batches)):
      loss, batch_prediction_probs = sess.run(
        [cost, prediction_probs],
        feed_dict={input_placeholder: test_batched_input_vecs[i],
                   input_length_placeholder: test_batched_input_lengths[i],
                   question_placeholder: test_batched_question_vecs[i],
                   question_length_placeholder: test_batched_question_lengths[i],
                   labels_placeholder: test_batched_answer_vecs[i]})

      total_validation_loss = total_validation_loss + loss

      batch_accuracy = np.equal(np.argmax(batch_prediction_probs, axis=1),
                                np.argmax(test_batched_answer_vecs[i], axis=1)).mean()

      sum_test_accuracy += batch_accuracy

    average_test_loss = total_test_loss / len(test_batches)
    test_accuracy = sum_test_accuracy / len(test_batches)

    print '=-=' * 5
    print 'Test accuracy: {}'.format(test_accuracy)
    print '=-=' * 5


if __name__ == "__main__":
  run_baseline()
