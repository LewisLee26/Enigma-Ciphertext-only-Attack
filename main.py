import re
import numpy as np
import torch
import torch.nn as nn
import enigma
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

umlaut = ["ä", "ü", "ö"]
replaceUmlaut = ["ae", "ue", "oe"]
eszett = "ß"
replaceEszett = "ss"
regexEszett = re.compile('ß')

def standardize_text(text):
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    for i in range(len(umlaut)):
        regexUmlaut = re.compile(umlaut[i])
        text = regexUmlaut.sub(replaceUmlaut[i], text)
    text = regexEszett.sub(replaceEszett, text)
    text = re.sub('[^a-z]', '', text)
    return text

def str_to_tensor(inputStr):
    max_length = 1000
    result = np.zeros((max_length, 1, 26), dtype=int)
    for i, char in enumerate(inputStr):
        result[i, 0, ord(char) - ord('a')] = 1

    result_tensor = torch.tensor(result)
    return result_tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
        # altered code
        self.i2o.weight = nn.Parameter(torch.zeros(output_size, input_size + hidden_size))
        self.i2o.bias = nn.Parameter(torch.zeros(output_size))


    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        combined = combined.to(device)
        hidden = self.i2h(combined).clone()
        output = self.i2o(combined)
        # output = self.softmax(output)
        output = torch.sigmoid(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size).clone()
    
classes = ['Encrypted']
num_classes = len(classes)
hidden_size = 256
input_size = 26

model = RNN(input_size, hidden_size, num_classes).to(device)
checkpoint = torch.load('models/RNNmodel.pt')

model.load_state_dict(checkpoint, strict=False)

# show parameters 
# for parameter in model.parameters():
#     print(parameter)
#     print(parameter.size())

rotors = [enigma.Rotor(enigma.rotors['I'], 0),
          enigma.Rotor(enigma.rotors['II'], 0),
          enigma.Rotor(enigma.rotors['III'], 0),
          enigma.Rotor(enigma.rotors['IV'], 0),
          enigma.Rotor(enigma.rotors['V'], 0)]

plugboard1 = enigma.Plugboard()
reflector1  = enigma.Reflector(enigma.reflectors['A'])

# prediction, rotor slot 1, position, rotor slot 2, position, rotor slot 3, position 
top_combinations = []

german_text = 'Hallo in die Runde, unter r/tee wurde diese Woche eine große Revolution erfolgreich abgeschloßen. Ausgehend vom Podcast „Das Podcast Ufo" wurde der amerikanische T-Shirt (=Tee) Subreddit von der Hörerschaft überrannt und wurde jetzt tatsächlich auch durch einen Mod zu einem deutschen Tee-Subreddit.Es liegt jetzt an uns r/Tee tatsächlich auch mit Tee-Inhalten zu füllen. Bitte werdet doch fleißig Mitglied und bereichert uns mit Teeklutur!!!!'
german_text = standardize_text(german_text)
encrypted_text = enigma.enigma(german_text, rotors[0], rotors[1], rotors[2], plugboard1, reflector1)

rotor_slots = [rotors[2], rotors[1], rotors[0]]


top_combination = [1, []]


start_time = time.time()

for a in range (26):
    for b in range(26):
        for c in range (26):
            rotors[0].setRotorPosition(a)
            rotors[1].setRotorPosition(b)
            rotors[2].setRotorPosition(c)
            input_text = enigma.enigma(encrypted_text, rotors[0], rotors[1], rotors[2], plugboard1, reflector1)

            with torch.no_grad():
                hidden = model.init_hidden().to(device)
                input_tensor = str_to_tensor(input_text).to(device)

                for i in range(input_tensor.size()[0]):
                    if torch.count_nonzero(input_tensor[i]) == 0:
                        break
                    y_prediction, hidden = model(input_tensor[i], hidden)
                
                if float(y_prediction) < top_combination[0] and a + b + c != 0:
                    top_combination = [float(y_prediction), [a, b, c]]

                print(f'\r{a*26**2+b*26+c}/{26**3}',end='')

print(top_combination)

end_time = time.time()
print(end_time-start_time)

