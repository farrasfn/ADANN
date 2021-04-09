import tensorflow as tf

saved_model_dir = 'Models/SecondModel'

# Convert the model.
model = tf.saved_model.load(saved_model_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 25])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('Inference/model.tflite', 'wb') as f:
  f.write(tflite_model)