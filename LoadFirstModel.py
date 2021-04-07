import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras


model = keras.models.load_model('Models/FirstModel')
dftest = pd.read_csv("Data/test.csv")
testfeatures = dftest.copy()
testlabels = testfeatures.pop('target')
testfeatures = np.array(testfeatures)

CATEGORIES = ["MildDemented","NonDemented"]
prediction = model.predict(testfeatures)
print(prediction)
#print (CATEGORIES[int(prediction[1][0])])