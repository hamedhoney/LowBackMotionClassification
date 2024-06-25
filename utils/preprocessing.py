import json
from typing import Optional
import numpy as np
from keras import utils
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load():
    with open('./output/AnamolySignals.json', 'r') as f:
        motionData = json.load(f)
        motionAssessments = [s['events'][0]['assessments'] for s in motionData]
        lowback = [m["trials"] for assess in motionAssessments for m in assess if "Low Back" in m["name"]]
        anamoly = [trial['motionResult']['position'] for sub in lowback for trial in sub]
        anamoly_labels = np.zeros((len(anamoly)), dtype=np.int16).tolist()
    with open('./output/NormalSignals.json', 'r') as f:
        motionData = json.load(f)
        motionAssessments = [s['events'][0]['assessments'] for s in motionData]
        lowback = [m["trials"] for assess in motionAssessments for m in assess if "Low Back" in m["name"]]
        normal = [trial['motionResult']['position'] for sub in lowback for trial in sub]
        normal_labels = np.ones((len(normal)), dtype=np.int16).tolist()
    return anamoly, anamoly_labels, normal, normal_labels

def zeroPad(dataset, maxSize: Optional[int]=None):
    if not maxSize:
        maxSize = int(np.mean([len(s) for s in dataset]))
    return utils.pad_sequences(dataset, maxlen=maxSize, padding='post', dtype='float32'), maxSize

def normalize(normalDataset, anomDataset):
  flattened = np.hstack(normalDataset)
  minVal = np.min(flattened)
  maxVal = np.max(flattened)
  return [(d-minVal)/(maxVal-minVal) for d in normalDataset], [(d-minVal)/(maxVal-minVal) for d in anomDataset]

def getData():
    anamoly, anamoly_labels, normal, normal_labels= load()
    # Normalize normal Data to [0 1]
    normal, anamoly = normalize(normal, anamoly)

    maxSize=512 #Signal length to include
    normal, maxSize = zeroPad(normal, maxSize=maxSize)
    anamoly, maxSize = zeroPad(anamoly, maxSize=maxSize)
    normalTrainData, normalValData, normalTrainLabels, normalValLabels = train_test_split(normal, normal_labels, test_size=0.2, random_state=42)
    anamolyValData, anamolyTestData, anamolyValLabels, anamolyTestLabels = train_test_split(anamoly, anamoly_labels, test_size=0.5, random_state=42)

    trainData = tf.reshape(normalTrainData, (len(normalTrainData), maxSize,))
    valData = tf.reshape(np.concatenate((normalValData,anamolyValData), axis=0), (len(normalValData)+len(anamolyValData), maxSize, ))
    valLabels = np.concatenate((normalValLabels,anamolyValLabels), axis=0)
    testData = tf.reshape(anamolyTestData, (len(anamolyTestData), maxSize,))
    return trainData, valData, testData, normalTrainLabels, valLabels, anamolyTestLabels
