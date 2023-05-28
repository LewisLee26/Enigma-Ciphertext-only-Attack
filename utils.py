import torch
import re

alphabet = "abcdefghijklmnopqrstuvwxyz"
n_alphbet = len(alphabet)

umlaut = ["ä", "ü", "ö"]
replaceUmlaut = ["ae", "ue", "oe"]
eszett = "ß"
replaceEszett = "ss"
regexEszett = re.compile('ß')

def returnAlphabet():
    return alphabet

def standardizeText(text):
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    for i in range(len(umlaut)):
        regexUmlaut = re.compile(umlaut[i])
        text = regexUmlaut.sub(replaceUmlaut[i], text)
    text = regexEszett.sub(replaceEszett, text)
    text = re.sub('[^a-z]', '', text)
    return text

def letter_to_index(letter):
    return alphabet.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_alphbet)
    tensor[letter_to_index(letter)] = 1
    return tensor

# def line_to_tensor(line):
#     line = standardizeText(line)
#     tensor = torch.zeros(len(line), 1, n_alphbet)
#     for i, letter in enumerate(line):
#         tensor[i][0][letter_to_index(letter)] = 1
#     return tensor