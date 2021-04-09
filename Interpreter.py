import numpy as np
import tensorflow as tf
import pandas as pd
from numpy import genfromtxt

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="Inference/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
#print (input_shape)

input_data = genfromtxt('Data/GLCMTest.csv', delimiter=',', skip_header=1)
input_data = np.float32(input_data)
input_data = np.atleast_2d(input_data)
#print(input_data)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
cat=['Alzheimer = Positive','Alzheimer = Negative']
output_data = interpreter.get_tensor(output_details[0]['index'])
if (output_data >= 1):
    output_data = 1
print(cat[int(output_data)])