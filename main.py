import numpy as np
import tensorflow as tf
import keras


@tf.keras.utils.register_keras_serializable()
def split_chars(input_data):
    s = tf.strings.regex_replace(input_data, ' ', '')
    s = tf.strings.regex_replace(s, '', ' ')
    s = tf.strings.strip(s)
    s = tf.strings.split(s, sep=' ')
    return s


custom_objects = {"split_chars": split_chars}

with keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model('saved_model/my_model')

model.summary()

while True:
    sample_text = input("input text: ")
    predictions = model.predict(np.array([sample_text]))
    print(predictions)
