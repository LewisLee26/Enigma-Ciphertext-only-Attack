from datasets import load_dataset, DatasetDict
import re
import random
import pandas as pd
from tqdm import tqdm

import ctypes
import ctypes.util

enigma_lib = ctypes.CDLL(r'C:\Users\lewis\Documents\GitHub\Enigma-Ciphertext-only-Attack\enigmac\enigma.dll')  # Replace with the actual path to your shared library

# Define the argument and return types for the run_enigma function
enigma_lib.run_enigma.argtypes = [
    ctypes.c_int,                 # reflector
    ctypes.POINTER(ctypes.c_int), # wheel_order
    ctypes.c_char_p,              # ring_setting
    ctypes.c_char_p,              # wheel_pos
    ctypes.c_char_p,              # plugboard_pairs
    ctypes.c_uint,                # plaintextsize
    ctypes.c_char_p               # from
]
enigma_lib.run_enigma.restype = ctypes.c_char_p

def run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_str):
    return enigma_lib.run_enigma(
        reflector,
        (ctypes.c_int * len(wheel_order))(*wheel_order),
        ctypes.create_string_buffer(ring_setting.encode()),
        ctypes.create_string_buffer(wheel_pos.encode()),
        ctypes.create_string_buffer(plugboard_pairs.encode()),
        plaintextsize,
        ctypes.create_string_buffer(from_str.encode())
    ).decode()


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
wheels = [0, 1, 2, 3, 4]
reflectors = [0, 1, 2]
max_plugboard_pairs = 12
def encode_text(text):
    wheel_order = random.sample(wheels, 3)
    reflector = random.choice(reflectors)
    ring_setting = random.choice(alphabet) + random.choice(alphabet) + random.choice(alphabet)
    wheel_pos = random.choice(alphabet) + random.choice(alphabet) + random.choice(alphabet) 
    plugboard_pairs = "".join(random.sample(alphabet, random.randint(0,max_plugboard_pairs)*2))
    plaintextsize = len(text)

    result = run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, text)

    label = random.randint(0,1)
    plugboard_pairs = ""
    if not label:
        wheel_order = random.sample(wheels, 3)
        reflector = random.choice(reflectors)
        ring_setting = random.choice(alphabet) + random.choice(alphabet) + random.choice(alphabet)
        wheel_pos = random.choice(alphabet) + random.choice(alphabet) + random.choice(alphabet) 
        plugboard_pairs = "".join(random.sample(alphabet, max_plugboard_pairs*2))
        # plugboard_pairs = "".join(random.sample(alphabet, random.randint(0,max_plugboard_pairs)*2))

    result = run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, result)

    return result, label

regex = re.compile('[^a-zA-Z]')


# dataset = load_dataset("bookcorpus")
dataset = load_dataset("wikipedia", "20220301.en")


print(dataset['train'])
dataset['train'] = dataset['train'][:1000]

df = pd.DataFrame({'text': [], 'label':[]})

max_text_len = 512
texts = dataset['train']['text']
for text in tqdm(texts):
    for i in range(len(text)//max_text_len): 
        processed_text = regex.sub('', text[i*max_text_len:i*max_text_len+max_text_len]).upper()
        processed_text, label = encode_text(processed_text)
        df.loc[-1] = [processed_text, label]  # adding a row
        df.index = df.index + 1  # shifting index
        # df = df.sort_index()  # sorting by index

    processed_text = regex.sub('', text[i*max_text_len:i*max_text_len+len(text)%max_text_len]).upper()
    processed_text, label = encode_text(processed_text)
    df.loc[-1] = [processed_text, label]  # adding a row
    df.index = df.index + 1  # shifting index
    # df = df.sort_index()  # sorting by index
df = df.sort_index()  # sorting by index

print(df)


df.to_parquet(r"dataset/enigma_binary_classification_wiki_en_12_plugs.parquet")