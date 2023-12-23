from datasets import load_dataset, DatasetDict
import re
import random

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
max_plugsboard_pairs = 8
def encode_text(text):
    wheel_order = random.sample(wheels, 3)
    reflector = random.choice(reflectors)
    ring_setting = random.choice(alphabet) + random.choice(alphabet) + random.choice(alphabet)
    wheel_pos = random.choice(alphabet) + random.choice(alphabet) + random.choice(alphabet) 
    plugboard_pairs = "".join(random.sample(alphabet, random.randint(0,max_plugsboard_pairs)*2))
    plaintextsize = len(text)

    result = run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, text)

    label = random.randint(0,1)
    plugboard_pairs = ""
    if not label:
        wheel_order = random.sample(wheels, 3)
        reflector = random.choice(reflectors)
        ring_setting = random.choice(alphabet) + random.choice(alphabet) + random.choice(alphabet)
        wheel_pos = random.choice(alphabet) + random.choice(alphabet) + random.choice(alphabet) 
        plugboard_pairs = "".join(random.sample(alphabet, random.randint(0,max_plugsboard_pairs)*2))

    result = run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, result)

    return result, label

regex = re.compile('[^a-zA-Z]')

# removing non-alphabet characters
# converting to uppercase
# trimming to 512 characters
def preprocess(example):
    text = example["text"]
    
    # Check if text is not a string and convert it to a string
    if not isinstance(text, str):
        text = str(text)
    
    processed_text = regex.sub('', text).upper()[:512]
    processed_text, label = encode_text(processed_text)
    return {"text": [processed_text], "label": [label]}


dataset = load_dataset("bookcorpus")
dataset = dataset.map(preprocess, batched=True)

dataset = dataset['train'].train_test_split(test_size=0.1)

dataset.save_to_disk('dataset/enigma_binary_classification_en_8_plugs')
