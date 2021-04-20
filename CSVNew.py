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
from tensorflow.keras.layers.experimental import preprocessing

#https://www.tensorflow.org/tutorials/load_data/csv
#https://gogul.dev/software/first-neural-network-keras

x_train = pd.read_csv('Data/train.csv')
#print (x_train.head())
x_feat = x_train.copy()
x_labels = x_feat.pop('target')
x_feat = np.array(x_feat)
#print('\n=================TRAIN FEATURES=================\n')
#print(x_feat.shape)
#print('\n=================TRAIN FEATURES=================\n')

#normalize the data
normalize = preprocessing.Normalization()
normalize.adapt(x_feat)

def get_compiled_model():
    model = tf.keras.Sequential([
        normalize, #normalization is so important
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(13, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    return model

ADModel = get_compiled_model()
ADModel.fit(x_feat ,x_labels , epochs=1000)

x_test = pd.read_csv('Data/test.csv')
#print (x_test.head())
xt_feat = x_test.copy()
xt_labels = xt_feat.pop('target')
xt_feat = np.array(xt_feat)
#print('\n=================TEST  FEATURES=================\n')
#print(xt_feat.shape)
#print('\n=================TEST  FEATURES=================\n')
print ('EVALUATE')

result = ADModel.evaluate(xt_feat,xt_labels)
ADModel.summary()
ADModel.save('Models/TestSecondModel')