from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
from get_babi_data import get_task_6_train
from get_babi_data import get_task_6_test
from get_glove import load_glove_vectors

# Takes in the data set with (input, question, answer) tuplets and the dictionary of glove
# vectors and returns the word vectors for the input, question, and answer.
def format_data(data, glove_dict):
  text_arr = []
  question_arr = []
  answer_arr = []
  for (text, question, answer) in data:
    # convert word array to word vector array for text
    text_vec = []
    for word in text:
      if word in glove_dict:
        wordvec = glove_dict[word]
      else:
        wordvec = np.random.rand(1, WORD_VECTOR_LENGTH)[0]
        wordvec /= np.sum(wordvec)
      text_vec.append(wordvec)

    question_arr.append(text_vec)

    # convert word array to word vector array for question
    question_vec = []
    for word in question:
      if word in glove_dict:
        wordvec = glove_dict[word]
      else:
        wordvec = np.random.rand(1, WORD_VECTOR_LENGTH)[0]
        wordvec /= np.sum(wordvec)
      question_vec.append(wordvec)

    text_arr.append(question_vec)

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

def nn(X_train, y_train, X_test, y_test):
    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape

    batch_size = 32
    nb_epoch = 200
    data_augmentation = True

    model = Sequential()
    model.add(Dense(200, input_dim=22,activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit(X_train, y_train,
              nb_epoch=nb_epoch, batch_size=batch_size,
              validation_data=(X_test, y_test))

def main():
    print "test"
    train_data, train_labels = format_data()
    print "train data loaded"
    test_data, test_labels = readData('val.csv')
    print "test data loaded"


    X_train = np.array(train_data)
    y_train = np.array(train_labels)
    X_test = np.array(test_data)
    y_test = np.array(test_labels)

    nn(X_train, y_train, X_test, y_test)

main()

