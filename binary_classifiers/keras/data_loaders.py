import csv
import os
import sys
import numpy as np
import random

def loadEmails(labelsFilename, spamdirname):
  allData = []
  test = []
  train = []
  labels = []
  labels_test = []
  filenames = []
  filenames_test = []
  fileContents = []
  fileContents_test = []
  labelsFile = csv.DictReader(open(labelsFilename))
  count = 1
  for row in labelsFile:
    assign = random.randint(1, 10)
    if assign<6:
      catlabel = np.int32(row['label'])
      labels.insert(count,catlabel)
      emailFilename = str(row['filename'])
      filenames.insert(count,emailFilename)
      # aFile = open(dirname + '/'+emailFilename, 'r')
      aFile = open(spamdirname + '/'+emailFilename, 'r')
      aFileContents = aFile.read()
      fileContents.insert(count,str(aFileContents))
    else:
      catlabel = np.int32(row['label'])
      labels_test.insert(count,catlabel)
      emailFilename = str(row['filename'])
      filenames_test.insert(count,emailFilename)
      aFile = open(spamdirname + '/'+emailFilename, 'r')
      aFileContents = aFile.read()
      fileContents_test.insert(count,str(aFileContents))
    count +=1

  labelsNp_train = np.array(labels,dtype=np.int32)
  filenamesNp_train = np.array(filenames)
  fileContentsNp_train = np.array(fileContents)
  labelsNp_test = np.array(labels_test,dtype=np.int32)
  filenamesNp_test = np.array(filenames_test)
  fileContentsNp_test = np.array(fileContents_test)

  train.insert(1,labelsNp_train)
  train.insert(2,filenamesNp_train)
  train.insert(3,fileContentsNp_train)

  test.insert(1,labelsNp_test)
  test.insert(2,filenamesNp_test)
  test.insert(3,fileContentsNp_test)
  
  results = []

  results.insert(1,train)
  results.insert(2,test)

  return results


def loadAllEmails(labelsFilename, spamdirname):
  allData = []
  labels = []
  filenames = []
  fileContents = []
  labelsFile = csv.DictReader(open(labelsFilename))
  count = 1
  for row in labelsFile:
      catlabel = np.int32(row['label'])
      labels.insert(count,catlabel)
      emailFilename = str(row['filename'])
      filenames.insert(count,emailFilename)
      aFile = open(spamdirname + '/'+emailFilename, 'r')
      aFileContents = aFile.read()
      fileContents.insert(count,str(aFileContents))
      count +=1

  labelsNp = np.array(labels,dtype=np.int32)
  filenamesNp = np.array(filenames)
  fileContentsNp = np.array(fileContents)

  allData.insert(1,labelsNp)
  allData.insert(2,filenamesNp)
  allData.insert(3,fileContentsNp)

  return allData


