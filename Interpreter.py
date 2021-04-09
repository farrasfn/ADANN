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
alzcond = (cat[int(output_data)])

sname= 'Farras Nabil'
snick= 'Farras'
#SENDING EMAIL 
import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "farras.spyderboy@gmail.com"  # Enter your address
receiver_email = "farrasfn@student.ub.ac.id"  # Enter receiver address
password = input("Password : ")
message = """\
Subject: Alzheimer Detection Result : {}

Hello {},\n\nYour result from our Neural Network Diagnosis\n {} \nWith regards,\n\tADANN Admin""".format(sname,snick,alzcond)

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)
