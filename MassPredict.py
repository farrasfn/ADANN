import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from keras.models import Model, Sequential

x_test = pd.read_csv('Data/test.csv')
#print (x_test.head())
xt_feat = x_test.copy()
xt_labels = xt_feat.pop('target')
xt_feat = np.array(xt_feat)
model = keras.models.load_model('Models/SecondModel', compile=True)
predictions = model.predict(xt_feat)
classes = np.argmax(predictions, axis = 1)
print(classes)
