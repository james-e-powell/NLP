# LSTM for sequence classification in the IMDB dataset
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
import tensorflow as tf

from data_loaders import *
from scoring import *

learn = tf.contrib.learn

# load the dataset but only keep the top n words, zero the rest
top_words = 120000

MAX_DOCUMENT_LENGTH = 100
n_words = 0
embedding_vecor_length = 32

trainData = loadAllEmails('../SPAMTrain_fix.label', '../TRAINING_DEBUG')

nm_folds = 10
cv = StratifiedKFold(trainData[0], n_folds=nm_folds, random_state=1)

argcount = len(sys.argv)
if argcount<2:
  print('usage: python lstm_word.py csvfilename')
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

    X_train = sequence.pad_sequences(X_train, maxlen=MAX_DOCUMENT_LENGTH)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_DOCUMENT_LENGTH)

    # create the model
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=MAX_DOCUMENT_LENGTH))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    history = model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

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


