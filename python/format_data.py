import numpy as np
import random

WORD_VECTOR_LENGTH = 50
NUM_CLASSES = 2
TRAINING_SPLIT = 0.8
INCLUDE_PUNCTUATION = True

def split_training_data(train_total):
  # Set seed for consistent splitting
  random.seed(31415)
  np.random.seed(9265)

  np.random.shuffle(train_total)
  split_index = int(len(train_total) * TRAINING_SPLIT)

  train = train_total[:split_index]
  dev = train_total[split_index:]

  return train, dev


# Takes in the data set with (input, question, answer) tuplets and the dictionary of glove
# vectors and returns the word vectors for the input, question, and answer.
def format_data(data, glove_dict):
  text_arr = []
  question_arr = []
  answer_arr = []
  for (text, question, answer) in data:
    # convert word array to word vector array for text
    # TODO figure out if we should include periods
    # print "Start example"
    # print text
    # print question
    # print answer
    text_vec = []
    for word in text:
      word = word.lower()
      if word in glove_dict:
        wordvec = glove_dict[word]
      # else:
      #   print "UNSEEN WORD"
      #   wordvec = np.random.rand(1, WORD_VECTOR_LENGTH)[0]
      #   wordvec /= np.sum(wordvec)
      text_vec.append(wordvec)

    text_arr.append(text_vec)

    # convert word array to word vector array for question
    question_vec = []
    for word in question:
      word = word.lower()
      if word in glove_dict:
        wordvec = glove_dict[word]
      # else:
      #   print "UNSEEN WORD"
      #   wordvec = np.random.rand(1, WORD_VECTOR_LENGTH)[0]
      #   wordvec /= np.sum(wordvec)
      question_vec.append(wordvec)

    question_arr.append(question_vec)

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


def batch_data(data, batch_size):
  """ 
  Takes in a list of tuples of (input, question, answer) and returns a list of length 
  (number of examples) / batch_size where each element is a batch of batch_size questions.

  data: a list of tuples of (input, question, answer)
  batch_size: The number of examples in each batch

  """

  # Compute total number of batches for the data set
  num_batches = len(data) / batch_size

  batched_data = []
  for i in range(num_batches):
    # Get current batch
    current_batch = []

    for j in range(batch_size):
      if i * batch_size + j < len(data):
        current_batch.append(data[i * batch_size + j])

    batched_data.append(current_batch)

  return batched_data


def convert_to_vectors(batched_data, glove_dict, max_input_length, max_question_length):
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
    batch_input_vecs = np.zeros((max_input_length, len(batch), WORD_VECTOR_LENGTH))
    batch_input_lengths = np.zeros(len(batch))
    batch_question_vecs = np.zeros((max_question_length, len(batch), WORD_VECTOR_LENGTH))
    batch_question_lengths = np.zeros(len(batch))
    batch_answer_vecs = np.zeros((len(batch), NUM_CLASSES))

    for i in range(len(batch)):
      example = batch[i]
      input = example[0]
      question = example[1]
      answer = example[2]

      # Add input vectors
      for j in range(len(input)):
        if j >= max_input_length:
          continue
        word = input[j]

        word_vector = get_word_vector(word, glove_dict)

        # Set the jth word of the ith batch to be the word vector
        batch_input_vecs[j, i, :] = word_vector

      # Add input length
      batch_input_lengths[i] = len(input)

      # Add question vectors
      for j in range(len(question)):
        if j >= max_question_length:
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

def convert_to_vectors_with_sentences(batched_data, glove_dict, max_input_length, max_num_sentences, max_question_length):
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
  batched_end_of_sentences = []
  batched_num_sentences = []
  batched_question_vecs = []
  batched_question_lengths = []
  batched_answer_vecs = []

  for batch in batched_data:

    # Batch is a list of tuples of length BATCH_SIZE or less

    # Create an array to hold all of the word vectors for the batch
    batch_input_vecs = np.zeros((max_input_length, len(batch), WORD_VECTOR_LENGTH))
    batch_input_lengths = np.zeros(len(batch))
    batch_end_of_sentences = np.zeros((max_num_sentences, len(batch)))
    batch_num_sentences = np.zeros(len(batch))
    batch_question_vecs = np.zeros((max_question_length, len(batch), WORD_VECTOR_LENGTH))
    batch_question_lengths = np.zeros(len(batch))
    batch_answer_vecs = np.zeros((len(batch), NUM_CLASSES))

    for i in range(len(batch)):
      example = batch[i]
      input = example[0]
      question = example[1]
      answer = example[2]

      num_sentences = 0

      # Add input vectors
      for j in range(len(input)):
        if j >= max_input_length:
          continue
        word = input[j]

        word_vector = get_word_vector(word, glove_dict)

        # Set the jth word of the ith batch to be the word vector
        batch_input_vecs[j, i, :] = word_vector

        if word == ".":
          batch_end_of_sentences[num_sentences, i] = j
          num_sentences += 1

      # Add input length
      batch_input_lengths[i] = len(input)

      # Add number of sentences
      batch_num_sentences[i] = num_sentences

      # Add question vectors
      for j in range(len(question)):
        if j >= max_question_length:
          continue
        word = question[j]

        word_vector = get_word_vector(word, glove_dict)

        # Set the jth word of the ith batch to be the word vector
        batch_question_vecs[j, i, :] = word_vector

      # Add question length
      batch_question_lengths[i] = len(question)

      # Add answer vectors

      # Convert answer to a one hot vector
      if answer == 'yes':
        answer = np.array([1, 0])
        answer = answer.reshape((1, NUM_CLASSES))
      else:
        answer = np.array([0, 1])
        answer = answer.reshape((1, NUM_CLASSES))

      batch_answer_vecs[i, :] = answer

    batched_input_vecs.append(batch_input_vecs)
    batched_input_lengths.append(batch_input_lengths)
    batched_end_of_sentences.append(batch_end_of_sentences)
    batched_num_sentences.append(batch_num_sentences)
    batched_question_vecs.append(batch_question_vecs)
    batched_question_lengths.append(batch_question_lengths)
    batched_answer_vecs.append(batch_answer_vecs)

  return batched_input_vecs, batched_input_lengths, batched_end_of_sentences, batched_num_sentences, \
         batched_question_vecs, batched_question_lengths, batched_answer_vecs


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

