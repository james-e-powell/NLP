"""
This model is similar to one described in this paper:
   "Character-level Convolutional Networks for Text Classification"
   http://arxiv.org/abs/1509.01626

and is somewhat alternative to the Lua code from here:
   https://github.com/zhangxiangxiao/Crepe
"""

import sys
import csv
import re

from data_loaders import *
from scoring import *

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import tensorflow as tf
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold

import numpy as np

learn = tf.contrib.learn

MAX_DOCUMENT_LENGTH = 200
num_chars = 256

# Prepare training and testing data
trainData = loadAllEmails('../SPAMTrain_fix.label', '../TRAINING_DEBUG')

nm_folds = 10
cv = StratifiedKFold(trainData[0], n_folds=nm_folds, random_state=1)

argcount = len(sys.argv)
if argcount<2:
  print('usage: python cnn_char.py csvfilename')
  sys.exit(0)

arglist = sys.argv
resultsFilename = arglist[1]
resultsBuffer = ''

resultsBuffer += 'accuracy,size,correct,false,precision,recall,f1\n'

for train_index, test_index in cv:
    X_train, X_test  = trainData[2][train_index], trainData[2][test_index]
    y_train, y_test = trainData[0][train_index], trainData[0][test_index]

    #######################################################################
    # Process vocabulary
    char_processor = learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)

    X_train = np.array(list(char_processor.fit_transform(X_train)))
    X_test = np.array(list(char_processor.transform(X_test)))
 
    #######################################################################

    # Build model
    model = Sequential()
    # model.add(Embedding(num_chars, X_train.shape[0], input_length=MAX_DOCUMENT_LENGTH))
    model.add(Embedding(num_chars, X_train.shape[0], input_length=X_train.shape[1]))
    model.add(Conv1D(64, 3, activation='relu', input_shape=(MAX_DOCUMENT_LENGTH, 100)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    print(model.summary())

    history = model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
    # Final evaluation of the model
    acc_scores = model.evaluate(X_test, y_test, verbose=0)

    y_predicted = model.predict(X_test)
    correct_count = 0
    predictions = []
    for prediction in y_predicted:
      if prediction <0.5:
        predictions.append(int(0))
      else:
        predictions.append(int(1))

    count=0
    false_neg_count =0
    for prediction in predictions:
      if prediction == 0 and y_test[count]==1:
        false_neg_count +=1
      if prediction == y_test[count]:
        correct_count+=1
      count+=1

    print("Accuracy: %.2f%%" % (acc_scores[1]*100))

    avg_loss = sum(history.history['loss'])/len(history.history['loss'])
    avg_acc=sum(history.history['acc'])/len(history.history['acc'])

    precision,recall,f1,support = scores(y_test, predictions)

    resultsBuffer += str(acc_scores[1]) + ',' + str(len(X_test)) + ',' +  str(correct_count) + ',' + str(false_neg_count) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n'

saveResults(resultsBuffer, resultsFilename)


