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