import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score

def scores(y_test, y_predicted):

  precision, recall, fscore, support = score(y_test, y_predicted)

  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(fscore))
  print('support: {}'.format(support))

  bPrecis, bRecall, bFscore, bSupport = score(y_test, y_predicted, average='binary')
  print('Precision ' + str(bPrecis))
  print('Recall ' + str(bRecall))
  print('F1 score ' + str(bFscore))

  return bPrecis,bRecall,bFscore,bSupport
