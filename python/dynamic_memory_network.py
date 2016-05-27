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
NUM_CLASSES = 2
MAX_EPISODES = 3
MAX_INPUT_SENTENCES = 40
EARLY_STOPPING = 2
MAX_INPUT_LENGTH = 200
MAX_QUESTION_LENGTH = 20
l2 = 0.001

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
  end_of_sentences_placeholder = tf.placeholder(tf.int32, shape=[MAX_INPUT_SENTENCES * BATCH_SIZE])
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
  end_of_sentences_placeholder: A vector of length (MAX_INPUT_SENTENCES * BATCH_SIZE) with the index of the end of each sentence
                                for each example in the batch in the matrix with all the word states which has dimension
                                (MAX_INPUT_LENGTH*BATCH_SIZE) x INPUT_HIDDEN_SIZE.

  """

  # Get outputs after every word
  # Outputs is a list of length MAX_INPUT_LENGTH where each element is BATCH_SIZE x HIDDEN_SIZE
  outputs, state = RNN(input_placeholder, input_length_placeholder, HIDDEN_SIZE, WORD_VECTOR_LENGTH, MAX_INPUT_LENGTH)

  # Convert list of outputs into a tensor of dimension (MAX_INPUT_LENGTH*BATCH_SIZE) x INPUT_HIDDEN_SIZE
  # Each row of the matrix is the output state for a word for that element of the batch
  # Row number for word is word_number*batch_size + batch_element_number
  output_mat = tf.concat(0, outputs)

  print output_mat
  print end_of_sentences_placeholder

  # Only project the states at the end of each sentence
  sentence_representations_mat = tf.gather(output_mat, end_of_sentences_placeholder)
  print sentence_representations_mat

  sentence_representations = tf.reshape(sentence_representations_mat, [MAX_INPUT_SENTENCES, BATCH_SIZE, HIDDEN_SIZE])

  print sentence_representations

  return sentence_representations, output_mat


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
                          tf.abs(tf.sub(c_t, m_prev)),
                          tf.reshape(tf.reduce_sum(tf.mul(tf.matmul(c_t, W_b), q), 1), (BATCH_SIZE, 1)),
                          tf.reshape(tf.reduce_sum(tf.mul(tf.matmul(c_t, W_b), m_prev), 1), (BATCH_SIZE, 1))])

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

def compute_regularization_penalty():

  penalty = tf.zeros([1])

  trainables =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  # TODO figure out why the loop is needed and why we cant use tf.get_collection(tf.GraphKeys.WEIGHTS)

  for variable in trainables:

    print variable.name

    if "W" in variable.name or "Matrix" in variable.name:
      print "Adding regularization for ", variable.name
      penalty += tf.nn.l2_loss(variable)

  return penalty


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
    sentence_states, all_outputs = input_module(input_placeholder, input_length_placeholder,
                                                end_of_sentences_placeholder)

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
  cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(projections, labels_placeholder))

  l2_loss = compute_regularization_penalty()

  cost = cross_entropy_loss + l2*l2_loss

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

        # print "Train batch ", train_batches[i]
        # print "End of sentences ", train_batched_end_of_sentences[i]

        loss, _, batch_prediction_probs, input_outputs, sentence_states_out = sess.run(
          [cost, optimizer, prediction_probs, all_outputs, sentence_states],
          feed_dict={input_placeholder: train_batched_input_vecs[i],
                     input_length_placeholder: train_batched_input_lengths[i],
                     end_of_sentences_placeholder: train_batched_end_of_sentences[i],
                     num_sentences_placeholder: train_batched_num_sentences[i],
                     question_placeholder: train_batched_question_vecs[i],
                     question_length_placeholder: train_batched_question_lengths[i],
                     labels_placeholder: train_batched_answer_vecs[i]})

        # end_of_first_sentence_first_batch = train_batched_end_of_sentences[i][0,0]
        #
        # print "Index end of first sentence:", end_of_first_sentence_first_batch
        #
        # print "Shape input outputs", np.shape(input_outputs)
        # print "States at end of first sentence for first element of batch", input_outputs[end_of_first_sentence_first_batch, 0, :]
        # print "States at end of first sentence for first element of batch {}".format(sentence_states[0,0:])
        # print "Train batch number of sentences:", train_batched_num_sentences[i]

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

      outfile = './outputs/dmn/lr_' + str(LEARNING_RATE) + '_hs_' + str(HIDDEN_SIZE) +'_e_' + str(MAX_EPOCHS) + '.txt'
      f = open(outfile, "a")
      f.write('train_acc, ' + str(training_accuracy) + '\n')
      f.write('train_loss, ' + str(average_training_loss) + '\n')
      f.write('val_acc, ' + str(validation_accuracy) + '\n')
      f.write('val_loss, ' + str(average_validation_loss) + '\n')
      f.close()

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
