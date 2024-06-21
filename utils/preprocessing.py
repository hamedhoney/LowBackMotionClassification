import json
from typing import Optional
import numpy as np
from keras import utils

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
