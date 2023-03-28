import numpy as np
import tensorflow as tf
import keras
import enigma
from itertools import combinations
import time

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "y", "z"]


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

print()

encryptedText = input("input text: ")

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

bestPrediction = 100
rotorSet = []

for combo in combinations(rotors, 3):
    decryptedText = enigma.enigma(encryptedText, combo[0], combo[1], combo[2], plugboard1, reflector1)
    prediction = model.predict(np.array([decryptedText]))
    if prediction < bestPrediction:
        bestPrediction = prediction
        rotorSet = combo

rotorPosition1 = 0
rotorPosition2 = 0
rotorPosition3 = 0

for a in range(26):
    for b in range(26):
        for c in range(26):
            rotorSet[0].setRotorPosition(a)
            rotorSet[1].setRotorPosition(b)
            rotorSet[2].setRotorPosition(c)
            decryptedText = enigma.enigma(encryptedText, rotorSet[0], rotorSet[1], rotorSet[2], plugboard1, reflector1)
            prediction = model.predict(np.array([decryptedText]))
            if prediction < bestPrediction:
                bestPrediction = prediction
                rotorPosition1 = a
                rotorPosition2 = b
                rotorPosition3 = c
            print(a*26*26 + b*26 + c)

plugboardPairs = []

for a in range(26):
    for b in range(26):
        if a != b:
            plugboard1.addCharacterPair([alphabet[a], alphabet[b]])
            prediction = model.predict(np.array([decryptedText]))
            plugboard1.removeCharacterPair()
            if prediction < bestPrediction:
                bestPrediction = prediction
                plugboardPairs.append([alphabet[a], alphabet[b]])
            print(a*26 + b)

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


