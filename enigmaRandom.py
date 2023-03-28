import enigma
import random

rotor1 = enigma.Rotor([], 0)
rotor2 = enigma.Rotor([], 0)
rotor3 = enigma.Rotor([], 0)
rotor4 = enigma.Rotor([], 0)
rotor5 = enigma.Rotor([], 0)

rotors = [rotor1, rotor2, rotor3, rotor4, rotor5]

plugboard1 = enigma.Plugboard([])

reflector1 = enigma.Reflector([])

text = input("input text: ")

rotor1.randomizeCharacterPairs()
rotor2.randomizeCharacterPairs()
rotor3.randomizeCharacterPairs()
rotor4.randomizeCharacterPairs()
rotor5.randomizeCharacterPairs()

rotor1.setRotorPosition(random.randint(0, 25))
rotor2.setRotorPosition(random.randint(0, 25))
rotor3.setRotorPosition(random.randint(0, 25))

plugboard1.randomizeCharacterPairs()
reflector1.randomizeCharacterPairs()


print([rotor1.returnCharacterPairs(), rotor2.returnCharacterPairs(), rotor3.returnCharacterPairs(), rotor4.returnCharacterPairs(), rotor5.returnCharacterPairs()])

print([rotor1.returnRotorPosition(),rotor2.returnRotorPosition(),rotor3.returnRotorPosition()])

print(plugboard1.returnCharacterPairs())
print(reflector1.returnCharacterPairs())

print("output text: "+enigma.enigma(text, rotor1, rotor2, rotor3, plugboard1, reflector1))
