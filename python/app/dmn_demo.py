from flask import render_template
from app import app
from flask import request
import tensorflow as tf
import numpy as np
import re

from dmn_with_word_embedding import add_placeholders, convert_to_vectors, RNN, \
  input_module, question_module, episodic_memory_module, answer_module, answer_tokens_to_index, \
  compute_regularization_penalty

from format_data import split_training_data, format_data, batch_data, convert_to_indices

from get_babi_data import remove_long_sentences, get_task_train, get_task_test

from get_glove import load_glove_embedding

# Good task 1 path
PATH_TO_MODEL = "/Users/patrickchase/Documents/CS224D/finalProject/dynamic-memory-networks/data/weights/dmn_lr_0.001_r_0.001_hs_80_e_150_d_0.9_t_2_bs_100.weights"
MAX_INPUT_SENTENCES = 70
TASK = 2
BATCH_SIZE = 100
MAX_INPUT_LENGTH = 200
MAX_QUESTION_LENGTH = 20

print "Task", TASK

# Get train dataset for task
train_total = get_task_train(TASK)
train_total = remove_long_sentences(train_total, MAX_INPUT_SENTENCES)

train, validation = split_training_data(train_total)

# Get all tokens from answers in training
answer_to_index = answer_tokens_to_index(train_total)

print answer_to_index

number_of_answers = len(answer_to_index)

print number_of_answers

index_to_answer = inv_map = {v: k for k, v in answer_to_index.items()}

# Get word to glove vectors dictionary
word_to_index, embedding_mat = load_glove_embedding()

def initialize_word_vectors(shape, dtype):
  return embedding_mat

# Create L tensor from embedding_mat
with tf.variable_scope("Embedding") as scope:
  L = tf.get_variable("L", shape=np.shape(embedding_mat),
                      initializer=tf.random_uniform_initializer(minval=-np.sqrt(3), maxval=np.sqrt(3)))


# Add placeholders
input_placeholder, input_length_placeholder, end_of_sentences_placeholder, num_sentences_placeholder, question_placeholder, \
question_length_placeholder, labels_placeholder, dropout_placeholder = add_placeholders()

# Input module
sentence_states, all_outputs = input_module(input_placeholder, input_length_placeholder,
                                            end_of_sentences_placeholder, dropout_placeholder)

# Question module
question_state = question_module(question_placeholder, question_length_placeholder, dropout_placeholder)

# Episodic memory module
episodic_memory_state, gates_for_episodes = episodic_memory_module(sentence_states, num_sentences_placeholder, question_state)

# Answer module
projections = answer_module(episodic_memory_state, number_of_answers, dropout_placeholder)

prediction_probs = tf.nn.softmax(projections)

saver = tf.train.Saver()

session = tf.Session()

saver.restore(session, PATH_TO_MODEL)

# with tf.Session() as sess:
#   # Load weights for model
#   saver.restore(sess, PATH_TO_MODEL)

def run_dmn(input, question):

  if input == None or question == None:
    return None, None, None, None, None

  input_split = re.findall(r"[\w']+|[.,!?;]", input)
  print input_split
  question_split = re.findall(r"[\w']+|[.,!?;]", question)
  print question_split

  # Dummy answer must be in the dataset
  input_question_answer = (input_split, question_split, unicode('hallway'))

  print input_question_answer

  dummy_data = []

  for i in range(2*BATCH_SIZE):
    dummy_data.append(input_question_answer)

  # Convert data into a batch
  dummy_batch = batch_data(dummy_data, BATCH_SIZE)

  # Convert words into indices
  val_batched_input_vecs, val_batched_input_lengths, val_batched_end_of_sentences, val_batched_num_sentences, val_batched_question_vecs, \
  val_batched_question_lengths, val_batched_answers = convert_to_indices(dummy_batch,
                                                                         word_to_index,
                                                                         answer_to_index,
                                                                         MAX_INPUT_LENGTH,
                                                                         MAX_INPUT_SENTENCES,
                                                                         MAX_QUESTION_LENGTH)

  print "Running DMN"

  # Run dmn
  probs, episode_1_gates, episode_2_gates, episode_3_gates = session.run(
    [prediction_probs, gates_for_episodes[0],
     gates_for_episodes[1], gates_for_episodes[2]],
    feed_dict={input_placeholder: val_batched_input_vecs[0],
               input_length_placeholder: val_batched_input_lengths[0],
               end_of_sentences_placeholder: val_batched_end_of_sentences[0],
               num_sentences_placeholder: val_batched_num_sentences[0],
               question_placeholder: val_batched_question_vecs[0],
               question_length_placeholder: val_batched_question_lengths[0],
               dropout_placeholder: 1.0})

  print "DMN finished"

  answer_probs = probs[0]

  index_answer = np.argmax(answer_probs)

  # Convert answer into a word
  answer = index_to_answer[index_answer]

  print "Answer is", answer

  # Get sentences
  sentences = re.split(r"[.]+", input)
  sentences.pop()
  print "Sentences", sentences

  num_sentences = len(sentences)
  print num_sentences

  print episode_1_gates[0]
  print episode_2_gates[0]
  print episode_3_gates[0]

  # Get gates
  gates_1 = episode_1_gates[0][:num_sentences]
  gates_2 = episode_2_gates[0][:num_sentences]
  gates_3 = episode_3_gates[0][:num_sentences]

  print gates_1
  print gates_2
  print gates_3

  return answer, sentences, gates_1, gates_2, gates_3


@app.route('/')
@app.route('/dmn_demo')
def index():
  input = request.args.get('input')
  question = request.args.get('question')

  answer, sentences, episode_1_gates, episode_2_gates, episode_3_gates = run_dmn(input, question)

  return render_template("dmn_demo.html",
                         input=input,
                         question=question,
                         answer=answer,
                         sentences=sentences,
                         episode_1_gates=episode_1_gates,
                         episode_2_gates=episode_2_gates,
                         episode_3_gates=episode_3_gates)
