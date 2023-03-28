import numpy as np
import tensorflow as tf
import os
import re

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


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[^a-z]', '')


@tf.keras.utils.register_keras_serializable()
def split_chars(input_data):
    s = tf.strings.regex_replace(input_data, ' ', '')
    s = tf.strings.regex_replace(s, '', ' ')
    s = tf.strings.strip(s)
    s = tf.strings.split(s, sep=' ')
    return s


VOCAB_SIZE = 26
encoder = tf.keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    split=split_chars)
encoder.adapt(raw_train_ds.map(lambda text, label: text))

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(raw_train_ds,
                    epochs=10,
                    batch_size=32,
                    validation_data=raw_val_ds,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(raw_test_ds)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

model.save('saved_model/my_model')
