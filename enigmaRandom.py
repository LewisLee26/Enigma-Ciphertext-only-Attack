import enigma
import random

rotor1 = enigma.Rotor(enigma.rotors["I"], 0)
rotor2 = enigma.Rotor(enigma.rotors["II"], 0)
rotor3 = enigma.Rotor(enigma.rotors["III"], 0)
rotor4 = enigma.Rotor(enigma.rotors["IV"], 0)
rotor5 = enigma.Rotor(enigma.rotors["V"], 0)

rotors = [rotor1, rotor2, rotor3, rotor4, rotor5]
plugboard1 = enigma.Plugboard("")
reflector1 = enigma.Reflector(enigma.reflectors["A"])

text = input("input text: ")

rotor1.setRotorPosition(random.randint(0, 25))
rotor2.setRotorPosition(random.randint(0, 25))
rotor3.setRotorPosition(random.randint(0, 25))

print(plugboard1.returnCharacterPairs())

print("output text: "+enigma.enigma(text, rotor1, rotor2, rotor3, plugboard1, reflector1))
