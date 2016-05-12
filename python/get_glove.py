import csv
import os
import numpy as np
# code adapted from github/cgpotts/cs224u/

def load_glove_vectors():
	"""Loads in the glove vectors from data/glove.6B """

	glove_home = '../data/glove.6B'
	src_filename = os.path.join(glove_home, 'glove.6B.50d.txt')
	reader = csv.reader(open(src_filename), delimiter=' ', quoting=csv.QUOTE_NONE) 
	GLOVE = {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}

	return GLOVE