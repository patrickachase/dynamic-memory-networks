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
from format_data import split_training_data, format_data, batch_data, convert_to_vectors_with_sentences, get_word_vector
from random import shuffle
from params import parse_args

#### MODEL PARAMETERS ####

params = parse_args()

WORD_VECTOR_LENGTH = 50
VOCAB_LENGTH = 10000
NUM_CLASSES = 2
MAX_EPISODES = 3
MAX_INPUT_SENTENCES = 40
EARLY_STOPPING = 2
MAX_INPUT_LENGTH = 200
MAX_QUESTION_LENGTH = 20

LEARNING_RATE = params['LEARNING_RATE']
HIDDEN_SIZE = params['HIDDEN_SIZE']
ATTENTION_GATE_HIDDEN_SIZE = params['ATTENTION_GATE_HIDDEN_SIZE']
MAX_EPOCHS = params['MAX_EPOCHS']
REG = params['REG']
DROPOUT = params['DROPOUT']
OUT_DIR = params['OUT_DIR']
TASK = params['TASK']
UPDATE_LENGTH = params['UPDATE_LENGTH']
BATCH_SIZE = params['BATCH_SIZE']


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

  input_placeholder = tf.placeholder(tf.float32, shape=[MAX_INPUT_LENGTH, BATCH_SIZE, WORD_VECTOR_LENGTH])
  input_length_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
  end_of_sentences_placeholder = tf.placeholder(tf.int32, shape=[MAX_INPUT_SENTENCES, BATCH_SIZE])
  num_sentences_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
  question_placeholder = tf.placeholder(tf.float32, shape=[MAX_QUESTION_LENGTH, BATCH_SIZE, WORD_VECTOR_LENGTH])
  question_length_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
  labels_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
  return input_placeholder, input_length_placeholder, end_of_sentences_placeholder, num_sentences_placeholder, \
         question_placeholder, question_length_placeholder, labels_placeholder


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
  return output, state


def count_positive_and_negative(answer_vecs):
  num_positive = 0
  for answer_vec in answer_vecs:

    if answer_vec[0, 0] == 1:
      num_positive = num_positive + 1

  num_negative = len(answer_vecs) - num_positive

  return num_positive, num_negative


def input_module(input_placeholder, input_length_placeholder, end_of_sentences_placeholder):
  """
  Returns a matrix of size MAX_INPUT_SENTENCES x BATCH_SIZE x HIDDEN_SIZE with the hidden states for each sentence
  for each example and a matrix of size BATCH_SIZE with the number of sentences in each example

  X: Input is a MAX_INPUT_LENGTH X BATCH_SIZE X WORD_VECTOR_LENGTH matrix with the words for each example in the batch
  input_length_placeholder: Matrix of BATCH_SIZE x 1 with the number of words in each example
  index_end_of_sentences: A matrix of MAX_INPUT_SENTENCES x BATCH_SIZE with the index of the end of each sentence for each example

  """

  # Get outputs after every word
  # Outputs is a list of length MAX_INPUT_LENGTH where each element is BATCH_SIZE x HIDDEN_SIZE
  outputs, state = RNN(input_placeholder, input_length_placeholder, HIDDEN_SIZE, WORD_VECTOR_LENGTH, MAX_INPUT_LENGTH)

  # Convert list of outputs into a tensor of dimension MAX_INPUT_LENGTH x BATCH_SIZE x INPUT_HIDDEN_SIZE
  output_mat = tf.concat(0, outputs)

  # Only project the states at the end of each sentence
  # Sentences matrix is now MAX_INPUT_SENTENCES x BATCH_SIZE x HIDDEN_SIZE
  # TODO figure out if this is the best way to get the state at the end of each sentence
  sentence_representations_mat = tf.gather(output_mat, end_of_sentences_placeholder)

  return sentence_representations_mat


def question_module(question_placeholder, question_length_placeholder):
  """
  Returns a matrix of BATCH_SIZE x HIDDEN_SIZE with the hidden states for each question in the batch

  question_placeholder: Matrix of MAX_QUESTION_LENGTH X BATCH_SIZE X WORD_VECTOR_LENGTH matrix with the words for each question in the batch
  question_length_placeholder: Matrix of BATCH_SIZE x 1 with the number of words in each question

  """

  outputs, state = RNN(question_placeholder, question_length_placeholder, HIDDEN_SIZE, WORD_VECTOR_LENGTH,
                       MAX_QUESTION_LENGTH)

  return state


def episodic_memory_module(sentence_states, number_of_sentences, question_state):
  """
  Returns a matrix of size BATCH_SIZE x HIDDEN_SIZE with the memory state for each input and question in the batch

  sentence_states: Matrix of MAX_INPUT_SENTENCES x BATCH_SIZE x HIDDEN_SIZE with the states at the end of each sentence for each example in the batch
  number_of_sentences: Matrix of BATCH_SIZE x 1 with the number of sentences in each example
  question_state: A matrix of BATCH_SIZE x HIDDEN_SIZE with the hidden states for each question in the batch

  """

  # Split sentence states into a list of length MAX_INPUT_SENTENCES where each element is BATCH_SIZE x HIDDEN_SIZE
  sentence_states = tf.split(0, MAX_INPUT_SENTENCES, sentence_states)

  squeezed = []
  for i in range(len(sentence_states)):
    squeezed.append(tf.squeeze(sentence_states[i]))

  sentence_states = squeezed

  # Each element of memory states will be BATCH_SIZE x HIDDEN_SIZE
  memory_states = []

  q = question_state

  # Initialize first memory state to be the question state
  m = q

  # There is an episode e and a previous memory state m_prev for each pass through the data
  for i in range(MAX_EPISODES):

    m_prev = m

    # Initialize first hidden state for episode to be zeros
    # TODO figure out if this is the right thing to do
    h = tf.zeros([BATCH_SIZE, HIDDEN_SIZE])
    final_h = tf.zeros([BATCH_SIZE, HIDDEN_SIZE])

    # Loop over the sentences for each episode
    for j in range(MAX_INPUT_SENTENCES):
      c_t = sentence_states[j]

      # Set scope for all these operations to be the episode
      with tf.variable_scope("episode", reuse=True if (j > 0 or i > 0) else None):
        W_b = tf.get_variable("W_b", shape=(HIDDEN_SIZE, HIDDEN_SIZE))
        W_1 = tf.get_variable("W_1", shape=(7 * HIDDEN_SIZE + 2, ATTENTION_GATE_HIDDEN_SIZE))
        b_1 = tf.get_variable("b_1", shape=(1, ATTENTION_GATE_HIDDEN_SIZE))
        W_2 = tf.get_variable("W_2", shape=(ATTENTION_GATE_HIDDEN_SIZE, 1))
        b_2 = tf.get_variable("b_2", shape=(1, 1))
        gru_cell_episode = rnn_cell.GRUCell(num_units=HIDDEN_SIZE)

        # Compute z for each batch
        # Z is BATCH_SIZE x (7 * HIDDEN_SIZE + 2)
        Z = tf.concat(1, [c_t, m_prev, q, tf.mul(c_t, q), tf.mul(c_t, m_prev), tf.abs(tf.sub(c_t, q)),
                          tf.abs(tf.sub(c_t, m_prev)), tf.matmul(c_t, tf.matmul(W_b, tf.transpose(q))),
                          tf.matmul(c_t, tf.matmul(W_b, tf.transpose(m_prev)))])

        # Compute G
        attention_gate_hidden_state = tf.tanh(tf.add(tf.matmul(Z, W_1), b_1))

        # g is BATCH_SIZE x 1 where each value signifies the gate for sentence j for that batch
        g = tf.sigmoid(tf.add(tf.matmul(attention_gate_hidden_state, W_2), b_2))

        # Compute next hidden state
        h_prev = h

        output, gru_state = gru_cell_episode(c_t, h_prev)

        h = tf.mul(g, gru_state) + tf.mul(1 - g, h_prev)

        # TODO figure out if this works for batches of data

        # Compute indices to copy through for
        copy_cond = (j >= number_of_sentences)

        # Only keep non zero indices if the sentence is within the size of the input for that element of the batch
        h = tf.select(copy_cond, tf.zeros((BATCH_SIZE, HIDDEN_SIZE)), h)

        final_h = tf.select(copy_cond, final_h, h)

    # Episode state is the final hidden state after pass over the data
    e = final_h

    # Compute next m with previous m and episode
    with tf.variable_scope("memory", reuse=True if i > 0 else None):
      gru_cell_memory = rnn_cell.GRUCell(num_units=HIDDEN_SIZE)
      output, m = gru_cell_memory(e, m_prev)

  # Return final memory state
  return m


def answer_module(episodic_memory_states):
  """
  Returns a matrix of size BATCH_SIZE x NUM_CLASSES with the unscaled probabilities for each class

  episodic_memory_state: Matrix of BATCH_SIZE x HIDDEN_SIZE with the memory state for each input and question in the batch

  """

  with tf.variable_scope("answer_module"):
    W_out = tf.get_variable("W_out", shape=(HIDDEN_SIZE, NUM_CLASSES))
    b_out = tf.get_variable("b_out", shape=(1, NUM_CLASSES))

  projections = tf.matmul(episodic_memory_states, W_out) + b_out

  return projections


def get_end_of_sentences(words):
  end_of_sentences = []

  for i in range(len(words)):
    word = words[i]
    if word == ".":
      end_of_sentences.append(i)

  return end_of_sentences

def _copy_some_through(new_output, new_state):
  # Use broadcasting select to determine which values should get
  # the previous state & zero output, and which values should get
  # a calculated state & output.
  copy_cond = (time >= sequence_length)
  return ([math_ops.select(copy_cond, zero_output, new_output)]
          + [math_ops.select(copy_cond, old_s, new_s)
             for (old_s, new_s) in zip(state, new_state)])

def _maybe_copy_some_through():
  """Run RNN step.  Pass through either no or some past state."""
  new_output, new_state = call_cell()
  new_state = (
    list(_unpacked_state(new_state)) if state_is_tuple else [new_state])

  if len(state) != len(new_state):
    raise ValueError(
      "Input and output state tuple lengths do not match: %d vs. %d"
      % (len(state), len(new_state)))

  return control_flow_ops.cond(
    # if t < min_seq_len: calculate and return everything
    time < min_sequence_length, lambda: [new_output] + new_state,
    # else copy some of it through
    lambda: _copy_some_through(new_output, new_state))


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
  train_batches = batch_data(train, BATCH_SIZE)
  validation_batches = batch_data(validation, BATCH_SIZE)
  test_batches = batch_data(test, BATCH_SIZE)

  # Convert batches into vectors
  train_batched_input_vecs, train_batched_input_lengths, train_batched_end_of_sentences, train_batched_num_sentences, train_batched_question_vecs, \
  train_batched_question_lengths, train_batched_answer_vecs = convert_to_vectors_with_sentences(
    train_batches, glove_dict, MAX_INPUT_LENGTH, MAX_INPUT_SENTENCES, MAX_QUESTION_LENGTH)

  val_batched_input_vecs, val_batched_input_lengths, val_batched_end_of_sentences, val_batched_num_sentences, val_batched_question_vecs, \
  val_batched_question_lengths, val_batched_answer_vecs = convert_to_vectors_with_sentences(validation_batches,
                                                                                            glove_dict,
                                                                                            MAX_INPUT_LENGTH,
                                                                                            MAX_INPUT_SENTENCES,
                                                                                            MAX_QUESTION_LENGTH)

  test_batched_input_vecs, test_batched_input_lengths, test_batched_end_of_sentences, test_batched_num_sentences, test_batched_question_vecs, \
  test_batched_question_lengths, test_batched_answer_vecs = convert_to_vectors_with_sentences(
    test_batches, glove_dict, MAX_INPUT_LENGTH, MAX_INPUT_SENTENCES, MAX_QUESTION_LENGTH)

  # Print summary statistics
  print "Training samples: {}".format(len(train))
  print "Validation samples: {}".format(len(validation))
  print "Testing samples: {}".format(len(test))
  print "Batch size: {}".format(BATCH_SIZE)
  print "Training number of batches: {}".format(len(train_batches))
  print "Validation number of batches: {}".format(len(validation_batches))
  print "Test number of batches: {}".format(len(test_batches))

  # Add placeholders
  input_placeholder, input_length_placeholder, end_of_sentences_placeholder, num_sentences_placeholder, question_placeholder, \
  question_length_placeholder, labels_placeholder = add_placeholders()

  # Input module
  with tf.variable_scope("input"):
    sentence_states = input_module(input_placeholder, input_length_placeholder, end_of_sentences_placeholder)

  # Question module
  with tf.variable_scope("question"):
    question_state = question_module(question_placeholder, question_length_placeholder)

  # Episodic memory module
  with tf.variable_scope("episode"):
    episodic_memory_state = episodic_memory_module(sentence_states, num_sentences_placeholder, question_state)

  # Answer module
  with tf.variable_scope("answer"):
    projections = answer_module(episodic_memory_state)

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
                     end_of_sentences_placeholder: train_batched_end_of_sentences[i],
                     num_sentences_placeholder: train_batched_num_sentences[i],
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
                     end_of_sentences_placeholder: val_batched_end_of_sentences[i],
                     num_sentences_placeholder: val_batched_num_sentences[i],
                     question_placeholder: val_batched_question_vecs[i],
                     question_length_placeholder: val_batched_question_lengths[i],
                     labels_placeholder: val_batched_answer_vecs[i]})

        total_validation_loss += loss

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
                   end_of_sentences_placeholder: test_batched_end_of_sentences[i],
                   num_sentences_placeholder: test_batched_num_sentences[i],
                   question_placeholder: test_batched_question_vecs[i],
                   question_length_placeholder: test_batched_question_lengths[i],
                   labels_placeholder: test_batched_answer_vecs[i]})

      total_test_loss += loss

      batch_accuracy = np.equal(np.argmax(batch_prediction_probs, axis=1),
                                np.argmax(test_batched_answer_vecs[i], axis=1)).mean()

      sum_test_accuracy += batch_accuracy

    average_test_loss = total_test_loss / len(test_batches)
    test_accuracy = sum_test_accuracy / len(test_batches)

    print '=-=' * 5
    print 'Test accuracy: {}'.format(test_accuracy)
    print '=-=' * 5


    # TODO add input loop so we can test and debug with our own examples
    input = ""
    while input:
      # Run model

      input = raw_input('> ')

if __name__ == "__main__":
  run_baseline()
