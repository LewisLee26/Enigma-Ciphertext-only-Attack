import re
import numpy as np
import torch
import torch.nn as nn
import random

umlaut = ["ä", "ü", "ö"]
replaceUmlaut = ["ae", "ue", "oe"]
eszett = "ß"
replaceEszett = "ss"
regexEszett = re.compile('ß')

def standardizeText(text):
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    for i in range(len(umlaut)):
        regexUmlaut = re.compile(umlaut[i])
        text = regexUmlaut.sub(replaceUmlaut[i], text)
    text = regexEszett.sub(replaceEszett, text)
    text = re.sub('[^a-z]', '', text)
    return text

def char_to_tensor(text):
    text = standardizeText(text)
    max_length = 1000 # 2^15
    result = np.zeros((max_length), dtype=int)
    for i, char in enumerate(text):
        result[i] = ord(char) - ord('a')
    return torch.from_numpy(result.astype(np.float32))

class LogisticRegression(nn.Module):
  def __init__(self, n_input_features):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(n_input_features, 1)

  def forward(self, x):
    y_predicited = torch.sigmoid(self.linear(x))
    return y_predicited

n_features = 1000
model = LogisticRegression(n_features)
model.load_state_dict(torch.load('models/model.pt'))
model.eval()

german_text = 'Hallo in die Runde, unter r/tee wurde diese Woche eine große Revolution erfolgreich abgeschloßen. Ausgehend vom Podcast „Das Podcast Ufo" wurde der amerikanische T-Shirt (=Tee) Subreddit von der Hörerschaft überrannt und wurde jetzt tatsächlich auch durch einen Mod zu einem deutschen Tee-Subreddit.Es liegt jetzt an uns r/Tee tatsächlich auch mit Tee-Inhalten zu füllen. Bitte werdet doch fleißig Mitglied und bereichert uns mit Teeklutur!!!!'
rand_text = ''
alphabet = "abcdefghijklmnopqrstuvwxyz"
alphabet = list(alphabet)

for i in range(100):
   rand_text += random.choice(alphabet)

input_text = rand_text
prediction = model(char_to_tensor(input_text))
print(standardizeText(input_text))
print(prediction)
predicted_cls = prediction.round()
classes = ['decrypted', 'encrypted']
