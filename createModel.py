import os
# removing tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt


# plots a graph between any given metric vs epochs
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


# creating a regular expression to remove all non-letter characters
regex = re.compile('[^a-z]')

# Get the current working directory
cwd = os.getcwd()

# creating a vocabulary of all letters used within the dataset
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

# getting the datasets
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


# separating the input into a array of individual characters
@tf.keras.utils.register_keras_serializable()
def split_chars(input_data):
    s = tf.strings.regex_replace(input_data, ' ', '')
    s = tf.strings.regex_replace(s, '', ' ')
    s = tf.strings.strip(s)
    s = tf.strings.split(s, sep=' ')
    return s


# creating custom layer to change the text input into a vector
vocab_size = 26
encoder = tf.keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=vocab_size,
    split=split_chars)
encoder.adapt(raw_train_ds.map(lambda text, label: text))

# creating the layers in the model
# model = tf.keras.Sequential([
#     encoder,
#     tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1)
# ])

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# training the model
history = model.fit(raw_train_ds,
                    epochs=10,
                    batch_size=batch_size,
                    validation_data=raw_val_ds,
                    validation_steps=30)

# summary of the models performance
test_loss, test_acc = model.evaluate(raw_test_ds)

model.summary()

plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.show()

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

model.save('saved_model/my_model')
print("model saved")
