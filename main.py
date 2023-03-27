import numpy as np
import tensorflow as tf
import os
import re

import keras

regex = re.compile('[^a-z]')

cwd = os.getcwd()  # Get the current working directory (cwd)

decryptedText = ""
for filename in os.listdir(os.getcwd() + "/dataset/train/decryption/"):
    with open(os.path.join(os.getcwd() + "/dataset/train/decryption/", filename), 'r') as f:  # open in readonly mode
        decryptedText += f.read()

vocab = sorted(set(decryptedText))

char2idx = {char: i for i, char in enumerate(vocab)}
idx2char = np.array(vocab)
text_encoded = [char2idx[c] for c in decryptedText]
text_encoded = np.array(text_encoded)


def split_chars(input_data):
    s = tf.strings.regex_replace(input_data, ' ', '')
    s = tf.strings.regex_replace(s, '', ' ')
    s = tf.strings.strip(s)
    s = tf.strings.split(s, sep=' ')
    return s


def textToVector(text):
    vector = []
    for i in range(len(text)):
        vector.append(vocab.index(text[i]))
    return vector


def vectorToText(vector):
    text = ""
    for i in vector:
        text += vocab[i]
    return text


batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    cwd + '/dataset/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    cwd + '/dataset/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    cwd + '/dataset/test',
    batch_size=batch_size)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[^a-z]', '')


VOCAB_SIZE = 26
encoder = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=VOCAB_SIZE,
    split=split_chars)
encoder.adapt(raw_train_ds.map(lambda text, label: text))

print(np.array(encoder.get_vocabulary()))
print()
print(encoder(example)[:3].numpy())

