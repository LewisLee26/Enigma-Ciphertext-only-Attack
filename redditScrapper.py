from pmaw import PushshiftAPI
import praw
import re
import enigma
import random
import os
import math

path = os.getcwd()

rotor1 = enigma.Rotor(enigma.rotors["I"], 0)
rotor2 = enigma.Rotor(enigma.rotors["II"], 0)
rotor3 = enigma.Rotor(enigma.rotors["III"], 0)
rotor4 = enigma.Rotor(enigma.rotors["IV"], 0)
rotor5 = enigma.Rotor(enigma.rotors["V"], 0)

rotors = [rotor1, rotor2, rotor3, rotor4, rotor5]
plugboard1 = enigma.Plugboard("")
reflector1 = enigma.Reflector(enigma.reflectors["A"])


def encryptTextRandom(text):
    random.shuffle(rotors)
    rotors[0].setRotorPosition(random.randint(0,25))
    rotors[1].setRotorPosition(random.randint(0,25))
    rotors[2].setRotorPosition(random.randint(0,25))
    plugboard1.randomizeCharacterPairs()
    return enigma.enigma(text, rotors[0], rotors[1], rotors[2], plugboard1, reflector1)


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


reddit = praw.Reddit(
    client_id="-Acse7KhzRcSv0hEt6Znfg",
    client_secret="OdxeAEFzN5cvFWEIvBVXvEq-PyFRaw",
    user_agent="WebScrapper"
)

reddit.read_only = True

api = PushshiftAPI(praw=reddit)

subreddit = 'Kopiernudeln'

submissions = list(api.search_submissions(subreddit=subreddit, limit=None))
print(len(submissions))

trainSplit = 0.8
testSplit = 1 - trainSplit

for i in range(math.floor(len(submissions)*trainSplit)):
    if submissions[i]['selftext'] == '[removed]' or submissions[i]['selftext'] == '[deleted]':
        continue
    submissionText = standardizeText(submissions[i]['selftext'])
    with open(path+"/dataset/train/decrypted/" + str(i) + ".txt", "w") as f:
        f.write(submissionText)
    with open(path+"/dataset/train/encrypted/" + str(i) + ".txt", "w") as f:
        f.write(encryptTextRandom(submissionText))
    print(i, end="\r")

trainOffset = math.floor(len(submissions)*trainSplit)
for i in range(math.floor(len(submissions)*testSplit)):
    if submissions[i]['selftext'] == '[removed]' or submissions[i]['selftext'] == '[deleted]':
        continue
    submissionText = standardizeText(submissions[i]['selftext'])
    with open(path+"/dataset/test/decrypted/" + str(i) + ".txt", "w") as f:
        f.write(submissionText)
    with open(path+"/dataset/test/encrypted/" + str(i) + ".txt", "w") as f:
        f.write(encryptTextRandom(submissionText))
    print(i+trainOffset, end="\r")
