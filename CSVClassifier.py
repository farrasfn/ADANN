import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import tensorflow as tf
import pandas as pd
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

#read data w/ pandas
dftrain = pd.read_csv("Data/train.csv")
print(dftrain.head())
print(dftrain.dtypes)
dftrain['labels'] = pd.Categorical(dftrain['labels'])
dftrain['labels'] = dftrain.labels.cat.codes
print(dftrain.head())

 #test dataset
dftest = pd.read_csv("Data/test.csv")
print(dftest.head())
print(dftest.dtypes)
dftest['labels'] = pd.Categorical(dftest['labels'])
dftest['labels'] = dftest.labels.cat.codes
print(dftest.head())

#load data using tf.data.Dataset

traintarget = dftrain.pop('target')

trdataset = tf.data.Dataset.from_tensor_slices((dftrain.values, traintarget.values))

for feat, targ in trdataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

  tf.constant(dftrain['labels'])
  train_dataset = trdataset.shuffle(len(dftrain)).batch(1)

testtarget = dftest.pop('target')
testdataset = tf.data.Dataset.from_tensor_slices((dftest.values, testtarget.values))
for feat, targ in testdataset.take(5) :
  print ('Features: {}, Target: {}'.format(feat, targ))

  tf.constant(dftest['labels'])
  test_dataset = testdataset.shuffle(len(dftest)).batch(1)

#compiled model 
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

#fit the model
model = get_compiled_model()
model.fit(train_dataset, epochs=15)
#save the model
model.save('Models/FirstModel')
#evaluate the model
print ('EVALUATE')
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))

