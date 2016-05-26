import argparse

LEARNING_RATE = 0.001
REG = 0.001
DROPOUT = 0.3
MAX_EPOCHS = 100
INPUT_HIDDEN_SIZE = 50
QUESTION_HIDDEN_SIZE = 50
ANSWER_HIDDEN_SIZE = 50
BATCH_SIZE = 100

def parse_args():
  """
  Parses the command line input.

  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-lr', default=LEARNING_RATE, help='learning rate', type=float)
  parser.add_argument('-reg', default=REG, help='regularization', type=float)
  parser.add_argument('-epochs', default=MAX_EPOCHS, help='number of epochs', type=int)
  parser.add_argument('-dropout', default=DROPOUT, help='dropout rate', type=float)
  parser.add_argument('-input_hidden_size', default=INPUT_HIDDEN_SIZE, help='hidden size for input module', type=int)
  parser.add_argument('-question_hidden_size', default=QUESTION_HIDDEN_SIZE, help='hidden size for question module', type=int)
  parser.add_argument('-answer_hidden_size', default=ANSWER_HIDDEN_SIZE, help='hidden size for answer module', type=int)
  parser.add_argument('-batch_size', default=BATCH_SIZE, help='batch size', type=int)

  args = parser.parse_args()

  params = {
  'LEARNING_RATE': args.lr,
  'REG': args.reg,
  'MAX_EPOCHS': args.epochs,
  'DROPOUT': args.dropout,
  'INPUT_HIDDEN_SIZE': args.input_hidden_size,
  'QUESTION_HIDDEN_SIZE': args.question_hidden_size,
  'ANSWER_HIDDEN_SIZE': args.answer_hidden_size,
  'BATCH_SIZE': args.batch_size
  }

  return params