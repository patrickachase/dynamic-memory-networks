from get_babi_data import *
from itertools import groupby

train_data = get_task_train(3)

max_sentences = 0
max_words = 0

num_greater = 0

for vec in train_data:
  sentence_vec = [list(group) for k, group in groupby(vec[0], lambda x: x == ".") if not k]
  num_sentences = len(sentence_vec)
  if num_sentences > max_sentences:
    max_sentences = num_sentences

  if num_sentences > 70:
    num_greater += 1

  for sentence in sentence_vec:
    if len(sentence) > max_words:
      max_words = len(sentence)


print "max sentences is: ", max_sentences
print "max words is: ", max_words
print "number of inputs with more than 70 sentences: ", num_greater









    