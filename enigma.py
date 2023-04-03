import random
import re

alphabet = "abcdefghijklmnopqrstuvwxyz"

# creating a regular expression to remove all non-letter characters
regex = re.compile('[^a-z]')

umlaut = ["ä", "ü", "ö"]
replaceUmlaut = ["ae", "ue", "oe"]
eszett = "ß"
replaceEszett = "ss"
regexEszett = re.compile('ß')


# the parent class for all parts of the enigma machine
class Connector:
    def __init__(self, characterPairs):
        self.characterPairs = characterPairs

    def setCharacterPairs(self, characterPairsInput):
        self.characterPairs = characterPairsInput

    def returnCharacterPairs(self):
        return self.characterPairs

    def clearCharacterPairs(self):
        self.characterPairs = []


# class for the rotor
class Rotor(Connector):
    def __init__(self, characterPairs, rotorPosition):
        super().__init__(characterPairs)
        self.rotorPosition = rotorPosition

    # creates a random set of rotor connections
    def randomizeCharacterPairs(self):
        self.characterPairs = ""
        alphabetCopy = alphabet.copy()
        while len(alphabetCopy) > 0:
            randomNumberPosition = random.randint(0, len(alphabetCopy) - 1)
            self.characterPairs.append(alphabetCopy[randomNumberPosition])
            del alphabetCopy[randomNumberPosition]

    # the boolean variable forward refers to which direction in it is passing through
    def returnCharacterPair(self, characterInput, Forward):
        if Forward:
            return self.characterPairs[(alphabet.index(characterInput) - self.rotorPosition) % 26]
        else:
            return alphabet[(self.characterPairs.index(characterInput) + self.rotorPosition) % 26]

    def returnRotorPosition(self):
        return self.rotorPosition

    def returnModRotorPosition(self):
        return self.rotorPosition % 26

    def setRotorPosition(self, inputNumber):
        self.rotorPosition = inputNumber

    def incrementRotorPosition(self):
        self.rotorPosition += 1


class Plugboard(Connector):
    def __init__(self, characterPairs):
        super().__init__(characterPairs)
        self.characterPairs = alphabet

    # creates a random number of random plugboard connections
    def randomizeCharacterPairs(self):
        self.characterPairs = ""
        alphabetCopy = alphabet
        characterPairs = alphabet
        for i in range(random.randint(0, len(alphabetCopy) / 2)):
            randomNumberPosition1 = random.randint(0, len(alphabetCopy) - 1)
            character1 = alphabetCopy[randomNumberPosition1]
            alphabetCopy = alphabetCopy[:randomNumberPosition1] + alphabetCopy[randomNumberPosition1 + 1:]
            randomNumberPosition2 = random.randint(0, len(alphabetCopy) - 1)
            character2 = alphabetCopy[randomNumberPosition2]
            alphabetCopy = alphabetCopy[:randomNumberPosition2] + alphabetCopy[randomNumberPosition2 + 1:]
            lst = list(characterPairs)
            lst[characterPairs.index(character1)], lst[characterPairs.index(character2)] = lst[characterPairs.index(
                character2)], lst[characterPairs.index(character1)]
            characterPairs = ''.join(lst)
        self.characterPairs = characterPairs

    def returnCharacterPair(self, characterInput):
        return self.characterPairs[alphabet.index(characterInput) % 26]

    def swapCharacters(self, characterPair):
        lst = list(self.characterPairs)
        lst[alphabet.index(characterPair[0])], lst[alphabet.index(characterPair[1])] = lst[alphabet.index(
            characterPair[1])], lst[alphabet.index(characterPair[0])]
        self.characterPairs = lst

    def setCharacterPairs(self, characterPairs):
        for i in characterPairs:
            self.swapCharacters(i)


class Reflector(Connector):
    def __init__(self, characterPairs):
        super().__init__(characterPairs)

    # creates a random set of rotor connections
    def randomizeCharacterPairs(self):
        self.characterPairs = ""
        alphabetCopy = alphabet
        characterPairs = alphabet
        for i in range(13):
            randomNumberPosition1 = random.randint(0, len(alphabetCopy) - 1)
            character1 = alphabetCopy[randomNumberPosition1]
            alphabetCopy = alphabetCopy[:randomNumberPosition1] + alphabetCopy[randomNumberPosition1 + 1:]
            randomNumberPosition2 = random.randint(0, len(alphabetCopy) - 1)
            character2 = alphabetCopy[randomNumberPosition2]
            alphabetCopy = alphabetCopy[:randomNumberPosition2] + alphabetCopy[randomNumberPosition2 + 1:]
            lst = list(characterPairs)
            lst[characterPairs.index(character1)], lst[characterPairs.index(character2)] = lst[characterPairs.index(
                character2)], lst[characterPairs.index(character1)]
            characterPairs = ''.join(lst)
        self.characterPairs = characterPairs

    def returnCharacterPair(self, characterInput):
        return self.characterPairs[alphabet.index(characterInput) % 26]


# the standard rotors used in WWII
# I-III is normal enigma, IV-V is army and VI-VIII is naval
rotors = {
    "I": "ekmflgdqvzntowyhxuspaibrcj",
    "II": "ajdksiruxblhwtmcqgznpyfvoe",
    "III": "bdfhjlcprtxvznyeiwgakmusqo",
    "IV": "esovpzjayquirhxlnftgkdcmwb",
    "V": "vzbrgityupsdnhlxawmjqofeck",
    "VI": "jpgvoumfyqbenhzrdkasxlictw",
    "VII": "nzjhgrcxmyswboufaivlpekqdt",
    "VII": "fkqhtlxocbjspdzramewniuygv"
}

# the standard reflectors used in WWII
reflectors = {
    "A": "ejmzalyxvbwfcrquontspikhgd",
    "B": "nzjhgrcxmyswboufaivlpekqdt",
    "C": "fkqhtlxocbjspdzramewniuygv",
}


# function to run the enigma machine
def enigma(inputText, rotorSlot1, rotorSlot2, rotorSlot3, plugboard, reflector):
    # changing the inputText into a form that can be processed by the enigma machine
    inputText = inputText.lower()
    for i in range(len(umlaut)):
        regexUmlaut = re.compile(umlaut[i])
        inputText = regexUmlaut.sub(replaceUmlaut[i], inputText)
    inputText = regexEszett.sub(replaceEszett, inputText)
    inputText = regex.sub('', inputText)

    # splitting the string into an array of characters
    arrayText = [char for char in inputText]

    outputArrayText = []
    for i in range(0, len(arrayText)):
        # one pass through the machine
        currentCharacter = arrayText[i]
        currentCharacter = plugboard.returnCharacterPair(arrayText[i])
        currentCharacter = rotorSlot3.returnCharacterPair(currentCharacter, True)
        currentCharacter = rotorSlot2.returnCharacterPair(currentCharacter, True)
        currentCharacter = rotorSlot1.returnCharacterPair(currentCharacter, True)
        currentCharacter = reflector.returnCharacterPair(currentCharacter)
        currentCharacter = rotorSlot1.returnCharacterPair(currentCharacter, False)
        currentCharacter = rotorSlot2.returnCharacterPair(currentCharacter, False)
        currentCharacter = rotorSlot3.returnCharacterPair(currentCharacter, False)
        currentCharacter = plugboard.returnCharacterPair(currentCharacter)

        outputArrayText.append(currentCharacter)

        # moves the rotor positions after each character has been processed
        rotorSlot1.incrementRotorPosition()
        if rotorSlot1.returnModRotorPosition() == 0:
            rotorSlot2.incrementRotorPosition()
            if rotorSlot2.returnModRotorPosition() == 0:
                rotorSlot3.incrementRotorPosition()

    # returning the final string connected together
    return ''.join(str(x) for x in outputArrayText)