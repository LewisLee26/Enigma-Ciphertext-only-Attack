import os

# removing tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import keras
import enigma
from itertools import permutations
import time

alphabet = "abcdefghijklmnopqrstuvwxyz"


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
rotor1 = enigma.Rotor(enigma.rotors["I"], 0)
rotor2 = enigma.Rotor(enigma.rotors["II"], 0)
rotor3 = enigma.Rotor(enigma.rotors["III"], 0)
rotor4 = enigma.Rotor(enigma.rotors["IV"], 0)
rotor5 = enigma.Rotor(enigma.rotors["V"], 0)
rotors = [rotor1, rotor2, rotor3, rotor4, rotor5]
plugboard1 = enigma.Plugboard("")
reflector1 = enigma.Reflector(enigma.reflectors["A"])

# finding the rotor settings
bestPrediction = 100
rotorSet = []

# for combo in permutations(rotors, 3):
#    combo[0].setRotorPosition(0)
#    combo[1].setRotorPosition(0)
#    combo[2].setRotorPosition(0)
#    decryptedText = enigma.enigma(encryptedText, combo[0], combo[1], combo[2], plugboard1, reflector1)
#    prediction = model.predict(np.array([decryptedText]))
#    if prediction < bestPrediction:
#        bestPrediction = prediction
#        rotorSet = combo

# combos = []
# for combo in permutations(rotors, 3):
#     combo[0].setRotorPosition(0)
#     combo[1].setRotorPosition(0)
#     combo[2].setRotorPosition(0)
#     decryptedText = enigma.enigma(encryptedText, combo[0], combo[1], combo[2], plugboard1, reflector1)
#     prediction = model.predict(np.array([decryptedText]))
#     combos.append([combo, prediction])

# combos = sorted(combos, key=lambda x: x[1])


rotorSet = [rotor1, rotor2, rotor3]

bestRotorPositions = []
for i in range(40):
    bestRotorPositions.append([[], 1])

for a in range(26):
    for b in range(26):
        for c in range(26):
            rotorSet[0].setRotorPosition(a)
            rotorSet[1].setRotorPosition(b)
            rotorSet[2].setRotorPosition(c)
            decryptedText = enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1)
            prediction = model.predict(np.array([decryptedText]))
            if prediction < bestRotorPositions[(len(bestRotorPositions) - 1)][1]:
                bestRotorPositions[(len(bestRotorPositions) - 1)] = [[a, b, c], prediction]
                bestRotorPositions = sorted(bestRotorPositions, key=lambda x: x[1])
            print(a * 26 * 26 + b * 26 + c)

print(rotorSet[0].returnRotorPosition())
print(rotorSet[1].returnRotorPosition())
print(rotorSet[2].returnRotorPosition())

# rotorPosition1 = 11
# rotorPosition2 = 17
# rotorPosition3 = 18
#
# rotorSet[0].setRotorPosition(rotorPosition1)
# rotorSet[1].setRotorPosition(rotorPosition2)
# rotorSet[2].setRotorPosition(rotorPosition3)
#
# print(enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1))
# bestPrediction = model.predict(
#     np.array([enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1)]))
# print("best prediction: " + str(bestPrediction))
#
# finding the plugboard settings
# plugboardPrediction = bestPrediction
# plugboardPair = ""
# plugboardPairs = []
# while True:
#     plugboardPair = ""
#     for a in range(26):
#         for b in range(26):
#             if a != b:
#                 if [alphabet[a]+alphabet[b]] in plugboardPairs:
#                     break
#                 rotorSet[0].setRotorPosition(rotorPosition1)
#                 rotorSet[1].setRotorPosition(rotorPosition2)
#                 rotorSet[2].setRotorPosition(rotorPosition3)
#                 plugboard1.swapCharacters(alphabet[a]+alphabet[b])
#                 decryptedText = enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1,
#                                               reflector1)
#                 plugboard1.swapCharacters(alphabet[a]+alphabet[b])
#                 prediction = model.predict(np.array([decryptedText]))
#                 if prediction < plugboardPrediction:
#                     plugboardPrediction = prediction
#                     plugboardPair = alphabet[a]+alphabet[b]
#                 # print(a*26 + b)
#                 print(prediction)
#         bestPrediction = plugboardPrediction
#     print(plugboardPair)
#     if plugboardPair:
#         plugboardPairs.append(plugboardPair)
#     else:
#         break

# displaying final information
end = time.time()
print("time: " + str(end - start))

for i in range(len(bestRotorPositions)):
    print(i)
    print(bestRotorPositions[i])

for i in bestRotorPositions:
    rotorSet[0].setRotorPosition(i[0][0])
    rotorSet[1].setRotorPosition(i[0][1])
    rotorSet[2].setRotorPosition(i[0][2])
    decryptedText = enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1)
    print(decryptedText)
