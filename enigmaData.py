import enigma
import random
import re

regex = re.compile('[^a-z]')

rotor1 = enigma.Rotor([],0)
rotor2 = enigma.Rotor([],0)
rotor3 = enigma.Rotor([],0)

plugboard1 = enigma.Plugboard([])

reflector1 = enigma.Reflector([])

def encryptText(i):
    rotor1.randomizeCharacterPairs()
    rotor1.setRotorPosition(random.randint(0,25))
    rotor2.randomizeCharacterPairs()
    rotor2.setRotorPosition(random.randint(0,25))
    rotor3.randomizeCharacterPairs()
    rotor3.setRotorPosition(random.randint(0,25))
    plugboard1.randomizeCharacterPairs()
    reflector1.randomizeCharacterPairs()
    print(i)
    return enigma.enigma(content[i*250:(i+1)*250],rotor1,rotor2,rotor3,plugboard1,reflector1)

with open("shakespeare.txt") as f:
    content = f.read()

f.close()

decryptedText = regex.sub('', content.lower())

with open("decrypted_train.txt", "w") as f:
    f.write(decryptedText[:3125000])

f.close()


with open("decrypted_test.txt", "w") as f:
    f.write(decryptedText[3125000:])

f.close()

output = ""
for i in range (12500):
    output += encryptText(i)

with open("encrypted_train.txt", "w") as f:
    f.write(output)

f.close()

output = ""
for i in range (2500 - 1):
    output += encryptText(i+12500)

with open("encrypted_test.txt", "w") as f:
    f.write(output)

f.close()

