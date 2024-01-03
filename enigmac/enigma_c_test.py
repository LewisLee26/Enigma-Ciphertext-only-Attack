from ctypes import *
from tqdm import tqdm

# Load the shared library
libenigma = CDLL(r'./enigmac/enigma.dll')

# Specify the argument types for the run_enigma function
libenigma.run_enigma.argtypes = [c_int, POINTER(c_int), POINTER(c_char), POINTER(c_char), POINTER(c_char), c_uint, POINTER(c_char)]

# Specify the return type
libenigma.run_enigma.restype = POINTER(c_char)

libenigma.free_memory.argtypes = [POINTER(c_char)]
libenigma.free_memory.restype = None

# Now you can call the function
reflector = 1
wheel_order = (c_int * 3)(1, 2, 3)
ring_setting = create_string_buffer(b"ABC")
wheel_pos = create_string_buffer(b"DEF")
plugboard_pairs = create_string_buffer(b"GHIJKL")
plaintextsize = 513
from_text = create_string_buffer(b"HELLO")

# result = libenigma.run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_text)

# result_bytes = string_at(result, plaintextsize)
# print(result_bytes.decode())

import random, string
import numpy as np
from itertools import permutations

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length)).upper().encode()

def tokenize(text):
    return np.array([ord(char) - ord("A") for char in list(text)], dtype=np.int64)

import onnx
import onnxruntime

path = r"model/model_1.onnx"

onnx_model = onnx.load(path)

ort_session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

# pbar = tqdm(total=(60*26**3))

wheels = [0, 1, 2, 3, 5]
combinations_list = list(permutations(wheels, 3))

from_text = create_string_buffer(randomword(512))


# for wheel_order in combinations_list:
#     for i in range(26):
#         for j in range(26):
#             for k in range(26):
#                 wheel_order = (c_int * 3)(wheel_order[0], wheel_order[1], wheel_order[2])
#                 result = libenigma.run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_text)

#                 # result_bytes = string_at(result, plaintextsize - 1)

#                 # inputs = {ort_session.get_inputs()[0].name: tokenize(result_bytes.decode())}
#                 # output = ort_session.run(None, inputs)

#                 pbar.update(1)

for i in tqdm(range(100000)):
    wheel_order = (c_int * 3)(1, 2, i%26)
    result = libenigma.run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_text)
    libenigma.free_memory(result)