import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

from data_loaders import *
from scoring import *

from sklearn.cross_validation import StratifiedKFold

import tensorflow as tf

learn = tf.contrib.learn

MAX_DOCUMENT_LENGTH = 150
n_words = 0

trainData = loadAllEmails('../SPAMTrain_fix.label', '../TRAINING_DEBUG')

nm_folds = 10
cv = StratifiedKFold(trainData[0], n_folds=nm_folds, random_state=1)

argcount = len(sys.argv)
if argcount<2:
  print('usage: python mlp_word.py csvfilename')
  sys.exit(0)

arglist = sys.argv
resultsFilename = arglist[1]
resultsBuffer = ''

resultsBuffer += 'accuracy,size,correct,false,precision,recall,f1\n'

for train_index, test_index in cv:
    X_train, X_test  = trainData[2][train_index], trainData[2][test_index]
    y_train, y_test = trainData[0][train_index], trainData[0][test_index]

    # Process vocabulary
    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    X_train = np.array(list(vocab_processor.fit_transform(X_train)))
    X_test = np.array(list(vocab_processor.transform(X_test)))
    n_words = len(vocab_processor.vocabulary_)
    # print('Total words: %d' % n_words)

    # create the model
    model = Sequential()
    model.add(Dense(256, input_dim=MAX_DOCUMENT_LENGTH, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    print(model.summary())

    history = model.fit(X_train, y_train,
          epochs=20,
          batch_size=25,
          verbose = 1)

    score = model.evaluate(X_test, y_test, batch_size=25)

    y_predicted = model.predict(X_test)
    count=0
    correct_count=0
    predicted_values = np.array(y_predicted, dtype=int)

    for prediction in y_predicted:
      if prediction == y_test[count]: 
        # print(str(prediction) + ' : ' + str(y_test[count]))
        correct_count+=1
      count+=1

    count=0
    false_neg_count =0
    for prediction in y_predicted:
      if prediction == 0 and y_test[count]==1:
        false_neg_count +=1
      count+=1

    avg_loss = sum(history.history['loss'])/len(history.history['loss'])
    avg_acc=sum(history.history['acc'])/len(history.history['acc'])

    precision, recall, f1, support = scores(y_test, predicted_values)

    resultsBuffer += str(avg_acc) + ',' + str(len(X_test)) + ',' +  str(correct_count) + ',' + str(false_neg_count) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n'

saveResults(resultsBuffer, resultsFilename)
