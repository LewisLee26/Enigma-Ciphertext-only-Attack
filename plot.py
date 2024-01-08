
import re
from tqdm.auto import tqdm
import numpy as np
import onnx
import onnxruntime
from itertools import permutations
from ctypes import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load the enigmac library
libenigma = CDLL(r'./enigmac/enigma.dll')
libenigma.run_enigma.argtypes = [c_int, POINTER(c_int), POINTER(c_char), POINTER(c_char), POINTER(c_char), c_uint, POINTER(c_char)]
libenigma.run_enigma.restype = POINTER(c_char)


# removing non-alphabet characters
# converting to uppercase
# trimming to 512 characters
regex = re.compile('[^a-zA-Z]')
def preprocess(text):
    return regex.sub('', text).upper()[:512]

def tokenize(text):
    tokens = np.array([ord(char) - ord("A") for char in list(text)], dtype=np.int32)
    if tokens.size < 512:
        zeros = np.zeros(512 - tokens.size, dtype=np.int32)
        return np.concatenate((tokens, zeros), dtype=np.int32)
    return tokens

# preset run_enigma arguments
reflector = 1
wheel_order = (c_int * 3)(0, 1, 2)
ring_setting = create_string_buffer(b"AAA")
wheel_pos = create_string_buffer(b"ANP")
plugboard_pairs = create_string_buffer(b"ARBYCOHXINMZ")
# plugboard_pairs = create_string_buffer(b"BYCOHXINMZ")
# plugboard_pairs = create_string_buffer(b"")
from_str = "CKFRKWZSEHCKSRFJIBWXRMMFHJCWJLFHFYNBWXULALKDVNLURSPWXNTBAWZKCQWVXCNCXXQVQDQLCAKYGSPIUQOUQXARYMHEIAVWBTZUZDYXZGHPGMHRUUWCELNZRJENVSDTFKMYXKOVZBQDEUZTFVZPLKTRJGLKBORCXYSLYMRAORDTIYDZSWAXTOSBJPINJPRZQNWECWNQOMKNGPCNRHWQAMGJXTLJHJNUJYYKTUSPRPTRALIZICFZJMKBFFQZPZGEBMUSIEJQVKGCTNFLZSEMHOSLDBYZJRYDRGQNJUPIAHJWZIXDADJMWQAGVJLGZGFCLMECEXBLRXTBCZIZVPCRPKUVGCXRJUFVBMEDIILDZAAYBFIREMHBHBZOWCRKQLYEKKGGVBQGRIATLOWOENQBBZRVIVTUTNNWRDTGFZCIABXVAZZPNLCTJKCJAEXVWHZWOEKCBQMKMSAWPIRCHXVJCMNFJFBAJKTNKLCMWBBYPDKTAVMCTBOXCHXSBQQYZIVQVCLQZQRFNXXUPOLQNMMBDGLRNHGVAOAPBUWBJMOZYXFGJURDETDCOAYDQQMNJLJZMXFVBJVKWVUJXTTBACBRIUJYBLCOZMOIRGRJLIZMPWKRJXUTTGVHRDZAKLSSIOIEHIYWLSQHCGHGRRUPICGHOJQSWGXYFFIBFKLLLRVJSTTZQWLJSWXLNRESBKXJKLZOBPRLQFZBPLZUPNPAUJFMVYVSCRCJRJHNKXUYPVQMWMWHNVGHPIZANQWUPAALEMHAYANFDUGMJDUVHRCDYPNBPOTKUOZYXHUXSLFMMRDLTLIXZGMVJPRYSYPTMNOZQUXNEOHZNNTGQEHALJHTWEHBQVKOOJTCGMSUXEHBOMXBXWUGLIALJPDBVMSJUZTUPYLOBOYUXXDGAUHYSNZAVSXJIEQVMFBNQZYXRASWFANPXKWSABNGEQPNHBFFNEXEONWAPVTMKQRABCIHJMPYCCMBVQNHMCHGNDKRCJWQIYJMBQGZCHCWVJPVWVMZENBRQXOKCAFPBGAKAEJZJJWDAZIJMVEOWLWMMSSDAMTKALHBFNEEVKXHDTVTKOHLRHVCFNEOXZKCLBLROFPHUNOYCRIWTPWJEKGCFVAWRQWFAYBXFPEWRGJMVSVFWPPUQYWWYLXLIZFXRKRTLGZPQTXDGQRTMKMDITHNCPIIDKTBJKCURTHAUITPIVDRXIWLIXXCDQHXREZZSCAGKIEUMJYEBGFFXXIDJAUNJPONFPLZCBONNJOUQEJIIPUSCBELPFJYVYJSVJXCYYLVLXUURRMPRBQHTRLRXOLSBMKDFSSGDWBFGKZUEJQRTBFVTOWPQMACUVVYAWZCMYQPOJGPEUAJYYGJRDPRGDYPVWGLQJVRLKOPBRAZOEXKGFNVYDDXYBVKWPELSPVPASQRQJECBUKHCTFXVNPTGUPGGOLLUZBPPPHLOCCPDGZUSDYRUCDUVRRELISSAQVVEHBYWVKILBRVNYSTKHTSRMPEEEJOBCIZVLTUQIKSODWZFDCFJODQPECXZTWWKJPSQDTCZPEWGIWCQWEFHGJPXIAAYTNTTVKOGFFCARLPNEAXNHGCTPNIVKYHIYMERGTGWOJCZFXYBYFCHMIOWLREWRPUYHRBQRDKXWVVRUUICXOACFKOZWTYWUULBKMQ"
# from_str = "NUWHAQRFMCBBSMYJKMEXGEXIYHXIKUSZBTUQADSRKGABHCMUMIESEPCGPWEAIJKHBZWRVLYVAQELXZJTWLQKIVORJGDAROMJTUQNXDGGVGUBPDWYFIUJQXVSBHXRLAMSAIJXCAONVDHKDYNZKYTRMIFLDNTXBEEJGAMSNYJUVFHWAUZSVNKSBXHFGGBUPOYBODZCLPQXWGIUMMOAGQCTISTDAYESSMQWUMUGYIUMYHBXBUJYQIDCHHTCMAZFZYGHJWLTCFUEJLFBRYKLGYZVLVEARZEESTUBKACDAXJIVULBZXACHGPXRCNAXIRTAFTBRBQFJOAXTLYBSRTJCZFUIXDKISLJNQVCWZCRGQOZJNZWVNLKBYBQFWHJALWDMKHQILXIEJQJUGSPWXEAPVJBKRBNTBPIORXFWLIVKKDCFXZAGDISWQQCVKZENNRMXTSHTPGKXOMXFGJQMFNRNDVQYMBFDSGJVPFRFLRGDHJNXEUPBKUPEOUXRDGRMTIWKWJS"
plaintextsize = len(preprocess(from_str))+1
from_str = create_string_buffer(preprocess(from_str).encode())


path = r"model/model_1_12.onnx"

onnx_model = onnx.load(path)

ort_session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


wheels = [0, 1, 2]
combinations_list = list(permutations(wheels, 3))
# combinations_list = [[0, 1, 2]]

def run_enigma_test():
    max_index = 17576*len(combinations_list)
    scores = np.zeros(17576*len(combinations_list))
    pbar = tqdm(total = max_index, desc="Searching for rotor")
    for wheel_order in combinations_list:
        for i in range(26):
            for j in range(26):
                for k in range(26):
                    wheel_order_2 = (c_int * 3)(wheel_order[0], wheel_order[1], wheel_order[2])
                    wheel_pos = create_string_buffer((chr(65 + i) + chr(65 + j) + chr(65 + k)).encode())

                    result = libenigma.run_enigma(reflector, wheel_order_2, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_str)
                    result_bytes = string_at(result, (plaintextsize-1))

                    inputs = {ort_session.get_inputs()[0].name: tokenize(result_bytes.decode())}
                    output = ort_session.run(None, inputs)
                    if wheel_order == (0, 1, 2) and i == 0 and j == 13 and k == 15:
                        # print(output)
                        correct_score = output
                        # print(result_bytes.decode())
                    scores[combinations_list.index(wheel_order)*17576+(i*26**2+j*26+k)] = output[0]

                    pbar.update(1)

    return scores, correct_score

scores, correct_score = run_enigma_test()

# plt.figure()

scores.sort()

# plt.plot(range(scores.size), scores, color='blue')

# plt.axhline(y=correct_score, color='black', linestyle='dotted', linewidth=1, label='Dotted Line')

# # Display the final plot
# plt.show()

index = np.where(scores == correct_score)

print(index[1])
print(index[1]/scores.size)
