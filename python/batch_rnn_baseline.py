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
EARLY_STOPPING = 2
MAX_INPUT_LENGTH = 200
MAX_QUESTION_LENGTH = 20
MAX_EPOCHS = 20
BATCH_SIZE = 100

# Number of training elements to train on before an update is printed
UPDATE_LENGTH = 1000


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


def RNN(X, num_words_in_X, hidden_size, max_input_size):

  # Split X into a list of tensors of length MAX_INPUT_LENGTH where each tensor is a 1xWORD_VECTOR_LENGTH vector
  # of the word vectors
  # TODO change input to be a list of tensors of length MAX_INPUT_LENGTH where each tensor is a BATCH_SIZExWORD_VECTOR_LENGTH vector
  X = tf.split(0, max_input_size, X)

  squeezed = []

  for i in range(len(X)):
    squeezed.append(tf.squeeze(X[i]))

  print "Length X: {}".format(len(X))

  gru_cell = rnn_cell.GRUCell(num_units=hidden_size, input_size=WORD_VECTOR_LENGTH)

  output, state = rnn.rnn(gru_cell, squeezed, sequence_length=num_words_in_X, dtype=tf.float32)

  print "State: {}".format(state)

  return output, state, X


# Takes in a list of tuples of (input, question, answer) and returns a list of length (number of examples) / BATCH_SIZE
# where each element is a batch of BATCH_SIZE questions
def batch_data(data):
  # Compute total number of batches for the data set
  num_batches = len(data) / BATCH_SIZE + 1

  batched_data = []

  for i in range(num_batches):

    # Get current batch
    current_batch = []

    for j in range(BATCH_SIZE):

      if i * BATCH_SIZE + j < len(data):
        current_batch.append(data[i * BATCH_SIZE + j])

    batched_data.append(current_batch)

  return batched_data


# Takes in a list of batches of data and converts them to a list of batched vectors
# Each element of the returned list contains all the vectors for a batch of data
# Output dimension is (max num words) x (BATCH_SIZE) x (WORD_VECTOR_LENGTH)
# If there are no fewer words than the max number of words zero vectors are added for padding
def convert_to_vectors(batched_data, glove_dict):
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
        word = input[j]

        word_vector = get_word_vector(word, glove_dict)

        # Set the jth word of the ith batch to be the word vector
        batch_input_vecs[j, i, :] = word_vector

      # Add input length
      batch_input_lengths[i] = len(input)

      # Add question vectors
      for j in range(len(question)):
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
    batched_question_vecs.append(batched_question_vecs)
    batched_question_lengths.append(batch_question_lengths)
    batched_answer_vecs.append(batch_answer_vecs)

  return batched_input_vecs, batched_input_lengths, batched_question_vecs, batched_question_lengths, batched_answer_vecs


def get_word_vector(word, glove_dict):
  if word in glove_dict:
    word_vec = glove_dict[word]
  else:
    word_vec = np.random.rand(1, WORD_VECTOR_LENGTH)[0]
    word_vec /= np.sum(word_vec)
  return word_vec


def answer_module(input_and_question):
  with tf.variable_scope("answer_module"):
    W_1 = tf.get_variable("W_1", shape=(INPUT_HIDDEN_SIZE + QUESTION_HIDDEN_SIZE, ANSWER_HIDDEN_SIZE))
    b_1 = tf.get_variable("b_1", shape=(1, ANSWER_HIDDEN_SIZE))

    W_out = tf.get_variable("W_out", shape=(ANSWER_HIDDEN_SIZE, NUM_CLASSES))
    b_out = tf.get_variable("b_out", shape=(1, NUM_CLASSES))

  h = tf.nn.relu(tf.matmul(input_and_question, W_1) + b_1)

  projections = tf.matmul(h, W_out) + b_out

  return projections


def run_baseline():
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
  batched_input_vecs, batched_input_lengths, batched_question_vecs, batched_question_lengths, batched_answer_vecs = convert_to_vectors(
    train_batches, glove_dict)

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
                                             MAX_INPUT_LENGTH)
  # Initialize question module
  with tf.variable_scope("question"):
    question_output, question_state, Q_input = RNN(question_placeholder, question_length_placeholder,
                                                   QUESTION_HIDDEN_SIZE, MAX_QUESTION_LENGTH)

  # Concatenate input and question vectors
  input_and_question = tf.concat(1, [input_state, question_state])

  # Answer model
  with tf.variable_scope("question"):
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
    best_loss = float('inf')
    best_val_epoch = 0

    sess.run(init)
    # train until we reach the maximum number of epochs
    for epoch in range(MAX_EPOCHS):

      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###

      # Shuffle training data
      # train_input_shuf = []
      # train_question_shuf = []
      # train_answer_shuf = []
      # index_shuf = range(len(text_train))
      # shuffle(index_shuf)
      # for i in index_shuf:
      #   train_input_shuf.append(text_train[i])
      #   train_question_shuf.append(question_train[i])
      #   train_answer_shuf.append(answer_train[i])
      #
      # text_train = train_input_shuf
      # question_train = train_question_shuf
      # answer_train = train_answer_shuf

      total_training_loss = 0
      sum_accuracy = 0

      prev_prediction = 0

      # Compute average loss on training data
      for i in range(len(train_batches)):

        # Print all inputs
        # print "Current input word vectors: {}".format(text_train[i])
        # print "Current number of words in input: {}".format(num_words_in_inputs)
        # print "Current question word vectors: {}".format(question_train[i])
        # print "Current number of words in question: {}".format(num_words_in_question)

        # print i
        # print num_words_in_inputs
        # print len(num_words_in_inputs)
        # print np.shape(num_words_in_inputs)
        loss, _, batch_prediction_probs, input_output_vec, input_state_vec, X_padded_input, question_output_vec, question_state_vec, X_padded_question, input_and_question_vec = sess.run(
          [cost, optimizer, prediction_probs, input_output[-1], input_state,
           X_input[-1], question_output[-1], question_state, Q_input[-1], input_and_question],
          feed_dict={input_placeholder: batched_input_vecs[i],
                     input_length_placeholder: batched_input_lengths[i],
                     question_placeholder: batched_question_vecs[i],
                     question_length_placeholder: batched_question_lengths[i],
                     labels_placeholder: batched_answer_vecs[i]})

        # Print all outputs and intermediate steps for debugging
        # print "Current input matrix with all words and padding: {}".format(X_input)
        # print "Current input matrix with all words and padding: {}".format(X_padded_input)
        # print "Current input matrix with all words and padding: {}".format(X_padded_question)
        # print "Current input ouput vector: {}".format(input_output_vec)
        # print "Current input state vector: {}".format(input_state_vec)
        # print "Current question ouput vector: {}".format(question_output_vec)
        # print "Current question state vector: {}".format(question_state_vec)
        # print "Current concatenated input and question embedding vector: {}".format(input_and_question_vec)

        # print "Current pred probs: {}".format(probs)
        # print "Current pred: {}".format(current_pred[0])
        # print "Current answer vector: {}".format(answer_train[i])
        # print "Current answer: {}".format(np.argmax(answer_train[i]))
        # print "Current loss: {}".format(loss)

        total_training_loss += loss

        batch_accuracy = np.equal(batch_prediction_probs, batched_answer_vecs[i]).mean()

        sum_accuracy += batch_accuracy

        print "Current average training loss: {}".format(total_training_loss / (i + 1))
        print "Current training accuracy: {}".format(sum_accuracy / (i + 1))

        # Print a training update
        if i % UPDATE_LENGTH == 0:
          print "Current average training loss: {}".format(total_training_loss / (i + 1))
          print "Current training accuracy: {}".format(sum_accuracy / (i + 1))
          # print "Current input matrix with all words and padding: {}".format(X_input)
          # print "Current input matrix with all words and padding: {}".format(X_padded_input)
          # print "Current input matrix with all words and padding: {}".format(X_padded_question)
          # print "Current input ouput vector: {}".format(input_output_vec)
          # print "Current input state vector: {}".format(input_state_vec)
          # print "Current question ouput vector: {}".format(question_output_vec)
          # print "Current question state vector: {}".format(question_state_vec)
          # print "Current concatenated input and question embedding vector: {}".format(input_and_question_vec)
          # print "Current W: {}".format(W_out_mat)
          # print "Current b: {}".format(b_out_mat)

        # Check if prediction changed
        # if prev_prediction != current_pred[0]:
        #   print "Prediction changed"

        prev_prediction = current_pred[0]

      average_training_loss = total_training_loss / len(train_batches)
      training_accuracy = sum_accuracy / len(train_batches)

      validation_loss = float('inf')

      total_validation_loss = 0
      num_correct_val = 0
      # Compute average loss on validation data
      # for i in range(len(validation)):
      #   num_words_in_inputs = [np.shape(text_val[i])[0]]
      #   num_words_in_question = [np.shape(question_val[i])[0]]
      #   loss, current_pred, probs, input_output_vec, input_state_vec, X_padded_input, question_output_vec, question_state_vec, X_padded_question, input_and_question_vec, W_out_mat, b_out_mat = sess.run(
      #     [cost, prediction, prediction_probs, input_output[num_words_in_inputs[0] - 1], input_state, X_input,
      #      question_output[num_words_in_question[0] - 1], question_state, Q_input, input_and_question, W_out, b_out],
      #     feed_dict={input_placeholder: text_val[i],
      #                input_length_placeholder: num_words_in_inputs,
      #                question_placeholder: question_val[i],
      #                question_length_placeholder: num_words_in_question,
      #                labels_placeholder: answer_val[i]})
      #
      #   if current_pred == np.argmax(answer_val[i]):
      #     num_correct_val = num_correct_val + 1
      #
      #   total_validation_loss = total_validation_loss + loss

      average_validation_loss = total_validation_loss / len(validation)
      validation_accuracy = float(num_correct_val) / len(validation)

      print 'Training loss: {}'.format(average_training_loss)
      print 'Training accuracy: {}'.format(training_accuracy)
      print 'Validation loss: {}'.format(average_validation_loss)
      print 'Validation accuracy: {}'.format(validation_accuracy)
      if average_validation_loss < best_loss:
        best_loss = average_validation_loss
        best_val_epoch = epoch
        saver.save(sess, '../data/weights/rnn.weights')
        print "Weights saved"
      # if epoch - best_val_epoch > EARLY_STOPPING:
      #   break
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
