import enigma
import random
import re
import os

path = os.getcwd()

regex = re.compile('[^a-z]')

rotor1 = enigma.Rotor(enigma.rotors["I"], 0)
rotor2 = enigma.Rotor(enigma.rotors["II"], 0)
rotor3 = enigma.Rotor(enigma.rotors["III"], 0)
rotor4 = enigma.Rotor(enigma.rotors["IV"], 0)
rotor5 = enigma.Rotor(enigma.rotors["V"], 0)

rotors = [rotor1, rotor2, rotor3, rotor4, rotor5]
plugboard1 = enigma.Plugboard("")
reflector1 = enigma.Reflector(enigma.reflectors["A"])


def encryptText(i):
    random.shuffle(rotors)
    rotors[0].setRotorPosition(random.randint(0,25))
    rotors[1].setRotorPosition(random.randint(0,25))
    rotors[2].setRotorPosition(random.randint(0,25))
    plugboard1.randomizeCharacterPairs()
    print(i)
    return enigma.enigma(content[i * size:(i + 1) * size], rotors[0], rotors[1], rotors[2], plugboard1, reflector1)

with open("shakespeare.txt") as f:
    content = f.read()

f.close()

decryptedText = regex.sub('', content.lower())
size = 1000
trainSplit = 0.7
testSplit = 1 - trainSplit

# decrypted_train
for i in range(int(((len(decryptedText)*trainSplit) // size) - 1)):
    with open(path+"/dataset/train/decryption/" + str(i) + ".txt", "w") as f:
        f.write(decryptedText[i * size:(i + 1) * size])
    f.close()
    print(i)

# decrypted_test
for i in range(int(((len(decryptedText)*testSplit)// size) - 1)):
    with open(path+"/dataset/test/decryption/" + str(i) + ".txt", "w") as f:
        f.write(decryptedText[int((i + (len(decryptedText)*testSplit // size) - 1) * size):int((i + (len(decryptedText)*testSplit // size)) * size)])
    f.close()
    print(i + len(decryptedText)*testSplit)

# encrypted_train
for i in range(int((len(decryptedText)*trainSplit)//size)):
    with open(path+"/dataset/train/encryption/" + str(i) + ".txt", "w") as f:
        f.write(encryptText(i))
    f.close()
    print(i)

# encrypted_test
for i in range(int(((len(decryptedText)*testSplit)//size)-1)):
    with open(path+"/dataset/test/encryption/" + str(i) + ".txt", "w") as f:
        f.write(encryptText(i + int(len(decryptedText)*testSplit)//size))
    f.close()
    print(i + (len(decryptedText)*testSplit)//size)
