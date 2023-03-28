import random
import re

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "y", "z"]
regex = re.compile('[^a-z]')


class Connector:
    def __init__(self, characterPairs):
        self.characterPairs = characterPairs

    def setCharacterPairs(self, characterPairsInput):
        self.characterPairs = characterPairsInput

    def returnCharacterPairs(self):
        return self.characterPairs

    def clearCharacterPairs(self):
        self.characterPairs = []


class Rotor(Connector):
    def __init__(self, characterPairs, rotorPosition):
        super().__init__(characterPairs)
        self.rotorPosition = rotorPosition

    def randomizeCharacterPairs(self):
        self.characterPairs = []
        alphabetCopy = alphabet.copy()
        while len(alphabetCopy) > 0:
            randomNumberPosition = random.randint(0, len(alphabetCopy) - 1)
            self.characterPairs.append(alphabetCopy[randomNumberPosition])
            del alphabetCopy[randomNumberPosition]

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

    def randomizeCharacterPairs(self):
        self.characterPairs = []
        alphabetCopy = alphabet.copy()
        numberOfPairs = random.randint(0, 13)
        for a in range(0, numberOfPairs):
            pair = []
            for b in range(0, 2):
                randomNumberPosition = random.randint(0, len(alphabetCopy) - 1)
                pair.append(alphabetCopy[randomNumberPosition])
                del alphabetCopy[randomNumberPosition]
            self.characterPairs.append(pair)

    def returnCharacterPair(self, characterInput):
        for i in range(0, len(self.characterPairs)):
            if self.characterPairs[i][0] == characterInput:
                return self.characterPairs[i][1]
            elif self.characterPairs[i][1] == characterInput:
                return self.characterPairs[i][0]
        return characterInput

    def addCharacterPair(self, characterPair):
        self.characterPairs.append(characterPair)

    def removeCharacterPair(self):
        del self.characterPairs[-1]


class Reflector(Plugboard):
    def __init__(self, characterPairs):
        super().__init__(characterPairs)

    def randomizeCharacterPairs(self):
        self.characterPairs = []
        alphabetCopy = alphabet.copy()
        for a in range(0, 13):
            pair = []
            for b in range(0, 2):
                randomNumberPosition = random.randint(0, len(alphabetCopy) - 1)
                pair.append(alphabetCopy[randomNumberPosition])
                del alphabetCopy[randomNumberPosition]
            self.characterPairs.append(pair)


def enigma(inputText, rotorSlot1, rotorSlot2, rotorSlot3, plugboard, reflector):
    inputText = regex.sub('', inputText.lower())
    arrayText = [char for char in inputText]

    outputArrayText = []
    for i in range(0, len(arrayText)):
        currentCharacter = []
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

        rotorSlot1.incrementRotorPosition()
        if rotorSlot1.returnModRotorPosition() == 0:
            rotorSlot2.incrementRotorPosition()
            if rotorSlot2.returnModRotorPosition() == 0:
                rotorSlot3.incrementRotorPosition()

    return ''.join(str(x) for x in outputArrayText)
