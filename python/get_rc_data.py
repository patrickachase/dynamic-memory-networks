import os


def parse_file(filename):
  """
  Takes path to a .question file
  Returns the context, question, and answer of the doc

  """
  text = [line.strip() for line in open(filename, 'r')]

  context = text[2].split(" ")
  question = text[4].split(" ")
  answer = text[6]

  return context, question, answer


def get_rc_dataset(path):
  """
  Takes path to folder containing all the question files to be parsed
  Returns a list of (input, question, answer) tuples

  """
  data = []
  for question_file in os.listdir(path):
    context, question, answer = parse_file(path + '/' + question_file)
    data.append((context, question, answer))

  return data


def get_rc_train():
  """
  Convenience function for getting the rc train dataset

  """
  return get_rc_dataset('../data/rc-data/cnn/questions/training')

def get_rc_val():
  """
  Convenience function for getting the rc val dataset

  """
  return get_rc_dataset('../data/rc-data/cnn/questions/validation')

def get_rc_test():
  """
  Convenience function for getting the rc test dataset

  """
  return get_rc_dataset('../data/rc-data/cnn/questions/test')

