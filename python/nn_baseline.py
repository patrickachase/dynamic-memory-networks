from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
from get_babi_data import get_task_6_train
from get_babi_data import get_task_6_test
from get_glove import load_glove_vectors
from format_data import split_training_data
from format_data import format_data

WORD_VECTOR_LENGTH = 50
NUM_CLASSES = 2

def sum_wordvecs(text_vec, question_vec):
    data = []

    for i in xrange(len(text_vec)):
        text_sum = np.zeros((1, WORD_VECTOR_LENGTH))
        for word in text_vec[i]:
            text_sum += word

        question_sum = np.zeros((1, WORD_VECTOR_LENGTH))
        for word in question_vec[i]:
            question_sum += word

        concat_vec = np.concatenate((text_sum, question_sum), 1)
        data.append(concat_vec[0])

    return data

def nn(X_train, y_train, X_test, y_test):
    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape

    batch_size = 32
    nb_epoch = 50
    data_augmentation = True

    model = Sequential()
    model.add(Dense(200, input_dim=2*WORD_VECTOR_LENGTH,activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X_train, y_train,
              nb_epoch=nb_epoch, batch_size=batch_size,
              validation_data=(X_test, y_test))

def main():
    # Get train dataset for task 6
    train_total = get_task_6_train()
    train, validation = split_training_data(train_total)

    # Get word to glove vectors dictionary
    glove_dict = load_glove_vectors()

    # Get data into word vector format
    text_train, question_train, train_labels = format_data(train, glove_dict)
    text_val, question_val, val_labels = format_data(validation, glove_dict)

    train_data = sum_wordvecs(text_train, question_train)
    val_data = sum_wordvecs(text_val, question_val)

    print "data loaded"

    X_train = np.array(train_data)
    y_train = np.array(train_labels)
    X_test = np.array(val_data)
    y_test = np.array(val_labels)

    nn(X_train, y_train, X_test, y_test)

main()

