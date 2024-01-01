import re
import ctypes
import ctypes.util
from tqdm.auto import tqdm
import concurrent.futures
import numpy as np
import onnx
import onnxruntime
from itertools import permutations

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


# removing non-alphabet characters
# converting to uppercase
# trimming to 512 characters
regex = re.compile('[^a-zA-Z]')
def preprocess(text):
    return regex.sub('', text).upper()[:512]

def tokenize(text):
    return np.array([ord(char) - ord("A") for char in list(text)], dtype=np.int64)

reflector = 1
wheel_order = [0, 1, 2]
ring_setting = "ABC"
ring_setting = "AAA"
wheel_pos = "AOR"
wheel_pos = "ANP"
plugboard_pairs = "ARBYCOHXINMZ"
plugboard_pairs = ""
plaintext = "INEVERREALLYEXPECTEDTOFINDMYSELFGIVINGADVICETOPEOPLEGRADUATINGFROMANESTABLISHMENTOFHIGHEREDUCATIONINEVERGRADUATEDFROMANYSUCHESTABLISHMENTINEVEREVENSTARTEDATONEIESCAPEDFROMSCHOOLASSOONASICOULDWHENTHEPROSPECTOFFOURMOREYEARSOFENFORCEDLEARNINGBEFOREIDBECOMETHEWRITERIWANTEDTOBEWASSTIFLINGIGOTOUTINTOTHEWORLDIWROTEANDIBECAMEABETTERWRITERTHEMOREIWROTEANDIWROTESOMEMOREANDNOBODYEVERSEEMEDTOMINDTHATIWASMAKINGITUPASIWENTALONGTHEYJUSTREADWHATIWROTEANDTHEYPAIDFORITORTHEYDIDNTANDOFTENTHEYCOMMISSIONEDMETOWRITESOMETHINGELSEFORTHEMWHICHHASLEFTMEWITHAHEALTHYRESPECTANDFONDNESSFORHIGHEREDUCATIONTHATTHOSEOFMYFRIENDSANDFAMILYWHOATTENDEDUNIVERSITIESWERECUREDOFLONGAGOLOOKINGBACKIVEHADAREMARKABLERIDEIMNOTSUREICANCALLITACAREERBECAUSEACAREERIMPLIESTHATIHADSOMEKINDOFCAREERPLANANDINEVERDIDTHENEARESTTHINGIHADWASALISTIMADEWHENIWAS15OFEVERYTHINGIWANTEDTODOTOWRITEANADULTNOVELACHILDRENSBOOKACOMICAMOVIERECORDANAUDIOBOOKWRITEANEPISODEOFDOCTORWHOANDSOONIDIDNTHAVEACAREERIJUSTDIDTHENEXTTHINGONTHELISTSOITHOUGHTIDTELLYOUEVERYTHINGIWISHIDKNOWNSTARTINGOUTANDAFEWTHINGSTHATLOOKINGBACKONITISUPPOSETHATIDIDKNOWANDTHATIWOULDALSOGIVEYOUTHEBESTPIECEOFADVICEIDEVERGOTWHICHICOMPLETELYFAILEDTOFOLLOWFIRSTOFALLWHENYOUSTARTOUTONACAREERINTHEARTSYOUHAVENOIDEAWHATYOUAREDOINGTHISISGREATPEOPLEWHOKNOWWHATTHEYAREDOINGKNOWTHERULESANDKNOWWHATISPOSSIBLEANDIMPOSSIBLEYOUDONOTANDYOUSHOULDNOTTHERULESONWHATISPOSSIBLEANDIMPOSSIBLEINTHEARTSWEREMADEBYPEOPLEWHOHADNOTTESTEDTHEBOUNDSOFTHEPOSSIBLEBYGOINGBEYONDTHEMANDYOUCANIFYOUDONTKNOWITSIMPOSSIBLEITSEASIERTODOANDBECAUSENOBODYSDONEITBEFORETHEYHAVENTMADEUPRULESTOSTOPANYONEDOINGTHATAGAINYETSECONDLYIFYOUHAVEANIDEAOFWHATYOUWANTTOMAKEWHATYOUWEREPUTHERETODOTHENJUSTGOANDDOTHAT"
from_str = "CKFRKWZSEHCKSRFJIBWXRMMFHJCWJLFHFYNBWXULALKDVNLURSPWXNTBAWZKCQWVXCNCXXQVQDQLCAKYGSPIUQOUQXARYMHEIAVWBTZUZDYXZGHPGMHRUUWCELNZRJENVSDTFKMYXKOVZBQDEUZTFVZPLKTRJGLKBORCXYSLYMRAORDTIYDZSWAXTOSBJPINJPRZQNWECWNQOMKNGPCNRHWQAMGJXTLJHJNUJYYKTUSPRPTRALIZICFZJMKBFFQZPZGEBMUSIEJQVKGCTNFLZSEMHOSLDBYZJRYDRGQNJUPIAHJWZIXDADJMWQAGVJLGZGFCLMECEXBLRXTBCZIZVPCRPKUVGCXRJUFVBMEDIILDZAAYBFIREMHBHBZOWCRKQLYEKKGGVBQGRIATLOWOENQBBZRVIVTUTNNWRDTGFZCIABXVAZZPNLCTJKCJAEXVWHZWOEKCBQMKMSAWPIRCHXVJCMNFJFBAJKTNKLCMWBBYPDKTAVMCTBOXCHXSBQQYZIVQVCLQZQRFNXXUPOLQNMMBDGLRNHGVAOAPBUWBJMOZYXFGJURDETDCOAYDQQMNJLJZMXFVBJVKWVUJXTTBACBRIUJYBLCOZMOIRGRJLIZMPWKRJXUTTGVHRDZAKLSSIOIEHIYWLSQHCGHGRRUPICGHOJQSWGXYFFIBFKLLLRVJSTTZQWLJSWXLNRESBKXJKLZOBPRLQFZBPLZUPNPAUJFMVYVSCRCJRJHNKXUYPVQMWMWHNVGHPIZANQWUPAALEMHAYANFDUGMJDUVHRCDYPNBPOTKUOZYXHUXSLFMMRDLTLIXZGMVJPRYSYPTMNOZQUXNEOHZNNTGQEHALJHTWEHBQVKOOJTCGMSUXEHBOMXBXWUGLIALJPDBVMSJUZTUPYLOBOYUXXDGAUHYSNZAVSXJIEQVMFBNQZYXRASWFANPXKWSABNGEQPNHBFFNEXEONWAPVTMKQRABCIHJMPYCCMBVQNHMCHGNDKRCJWQIYJMBQGZCHCWVJPVWVMZENBRQXOKCAFPBGAKAEJZJJWDAZIJMVEOWLWMMSSDAMTKALHBFNEEVKXHDTVTKOHLRHVCFNEOXZKCLBLROFPHUNOYCRIWTPWJEKGCFVAWRQWFAYBXFPEWRGJMVSVFWPPUQYWWYLXLIZFXRKRTLGZPQTXDGQRTMKMDITHNCPIIDKTBJKCURTHAUITPIVDRXIWLIXXCDQHXREZZSCAGKIEUMJYEBGFFXXIDJAUNJPONFPLZCBONNJOUQEJIIPUSCBELPFJYVYJSVJXCYYLVLXUURRMPRBQHTRLRXOLSBMKDFSSGDWBFGKZUEJQRTBFVTOWPQMACUVVYAWZCMYQPOJGPEUAJYYGJRDPRGDYPVWGLQJVRLKOPBRAZOEXKGFNVYDDXYBVKWPELSPVPASQRQJECBUKHCTFXVNPTGUPGGOLLUZBPPPHLOCCPDGZUSDYRUCDUVRRELISSAQVVEHBYWVKILBRVNYSTKHTSRMPEEEJOBCIZVLTUQIKSODWZFDCFJODQPECXZTWWKJPSQDTCZPEWGIWCQWEFHGJPXIAAYTNTTVKOGFFCARLPNEAXNHGCTPNIVKYHIYMERGTGWOJCZFXYBYFCHMIOWLREWRPUYHRBQRDKXWVVRUUICXOACFKOZWTYWUULBKMQ"
from_str = preprocess(from_str)
plaintextsize = len(from_str)



path = r"model/model_0.onnx"

onnx_model = onnx.load(path)

ort_session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


wheels = [0, 1, 2, 3, 5]
combinations_list = list(permutations(wheels, 3))


def run_enigma_test_thread(params):
    i, j, k, reflector, wheel_order, ring_setting, plugboard_pairs, plaintextsize, from_str = params
    wheel_pos = chr(65 + i) + chr(65 + j) + chr(65 + k)
    result = run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_str)
    
    inputs = {ort_session.get_inputs()[0].name: tokenize(result)}
    output = ort_session.run(None, inputs)
    
    return output[0], [i, j, k]

def run_enigma_test():
    max_index = 17576*60
    highest_val = 0
    highest_val_settings = []

    # params_list = [(i, j, k, reflector, wheel_order, ring_setting, plugboard_pairs, plaintextsize, from_str)
    #                for i in range(26) for j in range(26) for k in range(26) for wheel_order in combinations_list]
    pbar = tqdm(total = max_index)
    for wheel_order in combinations_list:
        
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     results = list(tqdm(executor.map(run_enigma_test_thread, params_list), total=max_index))
        for i in range(26):
            for j in range(26):
                for k in range(26):
                    wheel_pos = chr(65 + i) + chr(65 + j) + chr(65 + k)
                    try:
                        result = run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_str)
                    except Exception as e:
                        input(e)

                    inputs = {ort_session.get_inputs()[0].name: tokenize(result)}
                    output = ort_session.run(None, inputs)
    
                    if output[0] < highest_val:
                        highest_val = output[0]
                        highest_val_settings = [i, j, k]
                    # there is an error in enigmac
                    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe6 in position 11: invalid continuation byte
                    pbar.update(1)

        
    return highest_val, highest_val_settings

highest_val, highest_val_settings = run_enigma_test()

print(highest_val_settings)

wheel_pos = chr(65 + int(highest_val_settings[0])) + chr(65 + highest_val_settings[1]) + chr(65 + highest_val_settings[2])
result = run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_str)

print(result)

