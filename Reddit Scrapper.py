import requests
import pandas as pd
import enigmacpp
import random
import time
import pyarrow as pa
import pyarrow.parquet as pq
import utils


rotor1 = enigmacpp.Rotor(enigmacpp.rotors[0], 0)
rotor2 = enigmacpp.Rotor(enigmacpp.rotors[1], 0)
rotor3 = enigmacpp.Rotor(enigmacpp.rotors[2], 0)
rotor4 = enigmacpp.Rotor(enigmacpp.rotors[3], 0)
rotor5 = enigmacpp.Rotor(enigmacpp.rotors[4], 0)

rotors = [rotor1, rotor2, rotor3, rotor4, rotor5]
plugboard1 = enigmacpp.Plugboard()
reflector1 = enigmacpp.Reflector(enigmacpp.reflectors[0])


def encryptTextRandom(text):
    random.shuffle(rotors)
    rotors[0].setRotorPosition(random.randint(0,25))
    rotors[1].setRotorPosition(random.randint(0,25))
    rotors[2].setRotorPosition(random.randint(0,25))
    plugboard1.randomizeCharacterPairs()
    return enigmacpp.enigma(text, rotors[0], rotors[1], rotors[2], plugboard1, reflector1)

def parse(subreddit, after='', dataframe=pd.DataFrame):
    
    url_template = 'https://www.reddit.com/r/{}/top.json?t=all{}'
        
    headers = {
        'User-Agent': 'BlankBot'
    }

    params = f'&after={after}' if after else ''

    url = url_template.format(subreddit, params)
    response = requests.get(url, headers = headers)

    if response.ok:
        data = response.json()['data']
        for post in data['children']:
            pdata = post['data']
            post_id = pdata['id']
            if pdata['selftext'] == '[removed]' or pdata['selftext'] == '[deleted]' or pdata['selftext'] == '':
                continue
            selftext = pdata.get('selftext')
            print('ID: '+post_id)

            dataframe = pd.concat([dataframe, pd.DataFrame([{'Text':utils.standardizeText(selftext.strip()), 'Encryption':False}])], ignore_index=True)
            dataframe = pd.concat([dataframe, pd.DataFrame([{'Text':encryptTextRandom(utils.standardizeText(selftext.strip())), 'Encryption':True}])], ignore_index=True)

        return data['after'], dataframe
    else:
        print(f'Error {response.status_code}')
        return None

def main():
    start_time = time.time()

    df = pd.DataFrame(columns=['Text', 'Encryption'])

    subreddits = ['de', 'Austria','de_IAmA', 'duschgedanken', 'GuteNachrichten', 'HeuteLernteIch', 'Lagerfeuer', 
                 'Schreibkunst', 'Ratschlag', 'wortwitzkasse', 'WriteStreakGerman', 'buecher', 'wandern', 'radsport', 
                'Kampfsport', 'arbeitsleben', 'de_EDV', 'depression_de', 'einfach_posten', 'Eltern', 'egenbogen', 
                'Erasmus', 'Finanzen', 'germantrans', 'LegaladviceGerman', 'lehrerzimmer', 'recht', 'schwanger', 
                 'Weibsvolk', 'Kopiernudeln']
    
    for i in subreddits:
        print(i)
        after = ''
        while True:
            after, df = parse(i, after, df)
            if not after:
                break
        print('Dataframe Size: ' + str(df['Text'].size))
        print()
    
    end_time = time.time()
    print('Time: ' + str(end_time-start_time))

    print('Dataframe Size: ' + str(df['Text'].size))

    table = pa.Table.from_pandas(df)
    pq.write_table(table, 'dataset/germanEncryptionRaw.parquet')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Exiting...')