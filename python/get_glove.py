import csv
import os
import numpy as np
# code adapted from github/cgpotts/cs224u/

# Number of tokens in input glove vectors
NUM_TOKENS = 400000

def load_glove_vectors():
	"""Loads in the glove vectors from data/glove.6B """

	glove_home = '../data/glove.6B'
	src_filename = os.path.join(glove_home, 'glove.6B.50d.txt')
	reader = csv.reader(open(src_filename), delimiter=' ', quoting=csv.QUOTE_NONE) 
	GLOVE = {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}

	return GLOVE


def load_glove_embedding(dims=50):
  """Loads in the glove vectors from data/glove.6B """

  glove_home = '../data/glove.6B'
  src_filename = os.path.join(glove_home, 'glove.6B.' + str(dims) + 'd.txt')
  reader = csv.reader(open(src_filename), delimiter=' ', quoting=csv.QUOTE_NONE) 
  
  word_to_index = {}

  # Initialize embedding matrix to be all zeros with the correct dimension
  embedding_mat = np.zeros((NUM_TOKENS + 1, dims))

  counter = 0
  for line in reader:
    word_to_index[line[0]] = counter
    vec = np.array(list(map(float, line[1: ])))
    embedding_mat[counter] = vec
    counter += 1

  unk_vec = np.random.rand(dims)
  embedding_mat[counter] = unk_vec
  word_to_index['<unk>'] = counter

  return word_to_index, embedding_mat