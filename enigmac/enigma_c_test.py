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

# IOC functions
alph = "abcdefghijklmnopqrstuvwxyz"

def isLetter(char):
    return (char in alph)

def countLetters(text):
    count = 0
    for i in text:
        if(isLetter(i)):
            count += 1
    return count

def getIOC(text):
    letterCounts = []

    # Loop through each letter in the alphabet - count the number of times it appears
    for i in range(len(alph)):
        count = 0
        for j in text:
            if j == alph[i]:
                count += 1
        letterCounts.append(count)

    # Loop through all letter counts, applying the calculation (the sigma part)
    total = 0
    for i in range(len(letterCounts)):
        ni = letterCounts[i]
        total += ni * (ni - 1)

    N = countLetters(text)
    c = 26.0  # Number of letters in the alphabet
    total = float(total) / ((N * (N - 1)))
    return total

# setting for test
reflector = 1
wheel_order = [0, 1, 2]
ring_setting = "ABC"
ring_setting = "AAA"
wheel_pos = "AOR"
wheel_pos = "ANP"
# plugboard_pairs = "ARBYCOHXINMZ"
plugboard_pairs = ""
plaintextsize = 480
plaintext = "INEVERREALLYEXPECTEDTOFINDMYSELFGIVINGADVICETOPEOPLEGRADUATINGFROMANESTABLISHMENTOFHIGHEREDUCATIONINEVERGRADUATEDFROMANYSUCHESTABLISHMENTINEVEREVENSTARTEDATONEIESCAPEDFROMSCHOOLASSOONASICOULDWHENTHEPROSPECTOFFOURMOREYEARSOFENFORCEDLEARNINGBEFOREIDBECOMETHEWRITERIWANTEDTOBEWASSTIFLINGIGOTOUTINTOTHEWORLDIWROTEANDIBECAMEABETTERWRITERTHEMOREIWROTEANDIWROTESOMEMOREANDNOBODYEVERSEEMEDTOMINDTHATIWASMAKINGITUPASIWENTALONGTHEYJUSTREADWHATIWROTEANDTHEYPAIDFORITORTHEYDIDNTANDOFTENTHEYCOMMISSIONEDMETOWRITESOMETHINGELSEFORTHEMWHICHHASLEFTMEWITHAHEALTHYRESPECTANDFONDNESSFORHIGHEREDUCATIONTHATTHOSEOFMYFRIENDSANDFAMILYWHOATTENDEDUNIVERSITIESWERECUREDOFLONGAGOLOOKINGBACKIVEHADAREMARKABLERIDEIMNOTSUREICANCALLITACAREERBECAUSEACAREERIMPLIESTHATIHADSOMEKINDOFCAREERPLANANDINEVERDIDTHENEARESTTHINGIHADWASALISTIMADEWHENIWAS15OFEVERYTHINGIWANTEDTODOTOWRITEANADULTNOVELACHILDRENSBOOKACOMICAMOVIERECORDANAUDIOBOOKWRITEANEPISODEOFDOCTORWHOANDSOONIDIDNTHAVEACAREERIJUSTDIDTHENEXTTHINGONTHELISTSOITHOUGHTIDTELLYOUEVERYTHINGIWISHIDKNOWNSTARTINGOUTANDAFEWTHINGSTHATLOOKINGBACKONITISUPPOSETHATIDIDKNOWANDTHATIWOULDALSOGIVEYOUTHEBESTPIECEOFADVICEIDEVERGOTWHICHICOMPLETELYFAILEDTOFOLLOWFIRSTOFALLWHENYOUSTARTOUTONACAREERINTHEARTSYOUHAVENOIDEAWHATYOUAREDOINGTHISISGREATPEOPLEWHOKNOWWHATTHEYAREDOINGKNOWTHERULESANDKNOWWHATISPOSSIBLEANDIMPOSSIBLEYOUDONOTANDYOUSHOULDNOTTHERULESONWHATISPOSSIBLEANDIMPOSSIBLEINTHEARTSWEREMADEBYPEOPLEWHOHADNOTTESTEDTHEBOUNDSOFTHEPOSSIBLEBYGOINGBEYONDTHEMANDYOUCANIFYOUDONTKNOWITSIMPOSSIBLEITSEASIERTODOANDBECAUSENOBODYSDONEITBEFORETHEYHAVENTMADEUPRULESTOSTOPANYONEDOINGTHATAGAINYETSECONDLYIFYOUHAVEANIDEAOFWHATYOUWANTTOMAKEWHATYOUWEREPUTHERETODOTHENJUSTGOANDDOTHAT"
from_str = "CKFRKWZSEHCKSRFJIBWXRMMFHJCWJLFHFYNBWXULALKDVNLURSPWXNTBAWZKCQWVXCNCXXQVQDQLCAKYGSPIUQOUQXARYMHEIAVWBTZUZDYXZGHPGMHRUUWCELNZRJENVSDTFKMYXKOVZBQDEUZTFVZPLKTRJGLKBORCXYSLYMRAORDTIYDZSWAXTOSBJPINJPRZQNWECWNQOMKNGPCNRHWQAMGJXTLJHJNUJYYKTUSPRPTRALIZICFZJMKBFFQZPZGEBMUSIEJQVKGCTNFLZSEMHOSLDBYZJRYDRGQNJUPIAHJWZIXDADJMWQAGVJLGZGFCLMECEXBLRXTBCZIZVPCRPKUVGCXRJUFVBMEDIILDZAAYBFIREMHBHBZOWCRKQLYEKKGGVBQGRIATLOWOENQBBZRVIVTUTNNWRDTGFZCIABXVAZZPNLCTJKCJAEXVWHZWOEKCBQMKMSAWPIRCHXVJCMNFJFBAJKTNKLCMWBBYPDKTAVMCTBOXCHXSBQQYZIVQVCLQZQRFNXXUPOLQNMMBDGLRNHGVAOAPBUWBJMOZYXFGJURDETDCOAYDQQMNJLJZMXFVBJVKWVUJXTTBACBRIUJYBLCOZMOIRGRJLIZMPWKRJXUTTGVHRDZAKLSSIOIEHIYWLSQHCGHGRRUPICGHOJQSWGXYFFIBFKLLLRVJSTTZQWLJSWXLNRESBKXJKLZOBPRLQFZBPLZUPNPAUJFMVYVSCRCJRJHNKXUYPVQMWMWHNVGHPIZANQWUPAALEMHAYANFDUGMJDUVHRCDYPNBPOTKUOZYXHUXSLFMMRDLTLIXZGMVJPRYSYPTMNOZQUXNEOHZNNTGQEHALJHTWEHBQVKOOJTCGMSUXEHBOMXBXWUGLIALJPDBVMSJUZTUPYLOBOYUXXDGAUHYSNZAVSXJIEQVMFBNQZYXRASWFANPXKWSABNGEQPNHBFFNEXEONWAPVTMKQRABCIHJMPYCCMBVQNHMCHGNDKRCJWQIYJMBQGZCHCWVJPVWVMZENBRQXOKCAFPBGAKAEJZJJWDAZIJMVEOWLWMMSSDAMTKALHBFNEEVKXHDTVTKOHLRHVCFNEOXZKCLBLROFPHUNOYCRIWTPWJEKGCFVAWRQWFAYBXFPEWRGJMVSVFWPPUQYWWYLXLIZFXRKRTLGZPQTXDGQRTMKMDITHNCPIIDKTBJKCURTHAUITPIVDRXIWLIXXCDQHXREZZSCAGKIEUMJYEBGFFXXIDJAUNJPONFPLZCBONNJOUQEJIIPUSCBELPFJYVYJSVJXCYYLVLXUURRMPRBQHTRLRXOLSBMKDFSSGDWBFGKZUEJQRTBFVTOWPQMACUVVYAWZCMYQPOJGPEUAJYYGJRDPRGDYPVWGLQJVRLKOPBRAZOEXKGFNVYDDXYBVKWPELSPVPASQRQJECBUKHCTFXVNPTGUPGGOLLUZBPPPHLOCCPDGZUSDYRUCDUVRRELISSAQVVEHBYWVKILBRVNYSTKHTSRMPEEEJOBCIZVLTUQIKSODWZFDCFJODQPECXZTWWKJPSQDTCZPEWGIWCQWEFHGJPXIAAYTNTTVKOGFFCARLPNEAXNHGCTPNIVKYHIYMERGTGWOJCZFXYBYFCHMIOWLREWRPUYHRBQRDKXWVVRUUICXOACFKOZWTYWUULBKMQ"
plaintextsize = len(from_str)


result = run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_str)
print(result)

def run_enigma_test():
    count = 0
    highest_count = 0
    highest_pos = []
    for i in range(26):
        for j in range(26):
            for k in range(26):
                total_index = 676*i+26*j+k
                wheel_pos = chr(65+i) + chr(65+j) + chr(65+k)
                result = run_enigma(reflector, wheel_order, ring_setting, wheel_pos, plugboard_pairs, plaintextsize, from_str)
                # text = result.lower()
                # total = getIOC(text)
                count2 = 0
                for a, b in enumerate(result):
                    if b == plaintext[a]:
                        count2 += 1
                if count2 > highest_count:
                    highest_count = count2
                    highest_pos = [(i+1), (j+1), (k+1)]
                    highest_text = ""
                    for a, b in enumerate(result):
                        if b == plaintext[a]:
                            highest_text += b
                        else:
                            highest_text += "_"

                count += count2
    #             arr[total_index][0] = (total)
    #             arr[total_index][1], arr[total_index][2], arr[total_index][3] = (i+1), (j+1), (k+1)

    # sorted_indices = numpy.argsort(arr[:, 0])
    # sorted_arr_2d = arr[sorted_indices]

    # desc_sorted_arr = numpy.sort(arr, axis=0)[::-1]
    # print("First 5 elements:", sorted_arr_2d[::-1][:10])
    print(count/((26**3)*len(plaintext)))
    print(highest_count/len(plaintext))
    print(highest_pos)
    print(highest_text)

run_enigma_test()