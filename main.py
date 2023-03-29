import os
# removing tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import keras
import enigma
from itertools import combinations
import time

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "y", "z"]

# loading the text classification model
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

# the text to be decrypted
encryptedText = input("input text: ")

# set up for the enigma machine
start = time.time()
rotor1 = enigma.Rotor([], 0)
rotor2 = enigma.Rotor([], 0)
rotor3 = enigma.Rotor([], 0)
rotor4 = enigma.Rotor([], 0)
rotor5 = enigma.Rotor([], 0)
rotors = [rotor1, rotor2, rotor3, rotor4, rotor5]
plugboard1 = enigma.Plugboard([])
reflector1 = enigma.Reflector([])

rotorPairs = (eval(input("rotor pairs: ")))

for i in range(len(rotorPairs)):
    rotors[i].setCharacterPairs(rotorPairs[i])

reflector1.setCharacterPairs(eval(input("reflector1: ")))

# finding the rotor settings
bestPrediction = 100
rotorSet = []

# for combo in combinations(rotors, 3):
#    combo[0].setRotorPosition(0)
#    combo[1].setRotorPosition(0)
#    combo[2].setRotorPosition(0)
#    decryptedText = enigma.enigma(encryptedText, combo[0], combo[1], combo[2], plugboard1, reflector1)
#    prediction = model.predict(np.array([decryptedText]))
#    if prediction < bestPrediction:
#        bestPrediction = prediction
#        rotorSet = combo

# combos = []
# for combo in combinations(rotors, 3):
#     combo[0].setRotorPosition(0)
#     combo[1].setRotorPosition(0)
#     combo[2].setRotorPosition(0)
#     decryptedText = enigma.enigma(encryptedText, combo[0], combo[1], combo[2], plugboard1, reflector1)
#     prediction = model.predict(np.array([decryptedText]))
#     combos.append([combo, prediction])
#
# for a in combos:
#     print(a[1])
#
# combos = sorted(combos, key=lambda x: x[1])

rotorPosition1 = 0
rotorPosition2 = 0
rotorPosition3 = 0

rotorSet = [rotor1, rotor2, rotor3]

# for a in range(26):
#     for b in range(26):
#         for c in range(26):
#             rotorSet[0].setRotorPosition(a)
#             rotorSet[1].setRotorPosition(b)
#             rotorSet[2].setRotorPosition(c)
#             decryptedText = enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1)
#             prediction = model.predict(np.array([decryptedText]))
#             if prediction < bestPrediction:
#                 bestPrediction = prediction
#                 rotorPosition1 = a
#                 rotorPosition2 = b
#                 rotorPosition3 = c
#             print(a*26*26 + b*26 + c)

rotorPosition1 = 23
rotorPosition2 = 1
rotorPosition3 = 21

plugboardPairs = []

rotorSet[0].setRotorPosition(rotorPosition1)
rotorSet[1].setRotorPosition(rotorPosition2)
rotorSet[2].setRotorPosition(rotorPosition3)

print(enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1))
bestPrediction = model.predict(np.array([enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1)]))
print("best prediction: " + str(bestPrediction))


# finding the plugboard settings
plugboardPrediction = bestPrediction
for a in range(26):
    plugboardPair = []
    for b in range(26):
        if a != b:
            rotorSet[0].setRotorPosition(rotorPosition1)
            rotorSet[1].setRotorPosition(rotorPosition2)
            rotorSet[2].setRotorPosition(rotorPosition3)
            plugboard1.addCharacterPair([alphabet[a], alphabet[b]])
            decryptedText = enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1)
            print(plugboard1.returnCharacterPairs())
            plugboard1.removeCharacterPair()
            prediction = model.predict(np.array([decryptedText]))
            if prediction < plugboardPrediction:
                plugboardPrediction = prediction
                plugboardPair = [alphabet[a], alphabet[b]]
            # print(a*26 + b)

            print(prediction)
    bestPrediction = plugboardPrediction
    if plugboardPair:
        plugboardPairs.append(plugboardPair)

# displaying final information
end = time.time()
print("time: " + str(end - start))
print("prediction: " + str(bestPrediction))
print("rotors: " + str(rotorSet))
print("rotor 1 pos: " + str(rotorPosition1))
print("rotor 2 pos: " + str(rotorPosition2))
print("rotor 3 pos: " + str(rotorPosition3))
print("plugboard: " + str(plugboardPairs))
rotorSet[0].setRotorPosition(rotorPosition1)
rotorSet[1].setRotorPosition(rotorPosition2)
rotorSet[2].setRotorPosition(rotorPosition3)
plugboard1.setCharacterPairs(plugboardPairs)
print("output text: " + enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1))
