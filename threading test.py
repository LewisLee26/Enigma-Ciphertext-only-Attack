import enigmacpp
import threading
import multiprocessing
import time
from itertools import permutations
import numpy as np
import operator

start_time = time.time()
plugboard = enigmacpp.Plugboard(enigmacpp.alphabet)
reflector = enigmacpp.Connector(enigmacpp.reflectors[0])
test_string = "Hello World!"

def enigmaTest(slotPermutation):
    iteration_best = np.array([1])
    for rotorPosition1 in range(26):
        for rotorPosition2 in range(26):
            for rotorPosition3 in range(26):
                output = enigmacpp.enigma(test_string, 
                                        enigmacpp.Rotor(enigmacpp.rotors[slotPermutationsArray[slotPermutation][0]], rotorPosition1), 
                                        enigmacpp.Rotor(enigmacpp.rotors[slotPermutationsArray[slotPermutation][1]], rotorPosition2), 
                                        enigmacpp.Rotor(enigmacpp.rotors[slotPermutationsArray[slotPermutation][2]], rotorPosition3), 
                                        plugboard, 
                                        reflector)
                # run classification model here
                score = 0
                if score > iteration_best[0]:
                    iteration_best = [score,
                                    slotPermutationsArray[slotPermutation][0],
                                    slotPermutationsArray[slotPermutation][1],
                                    slotPermutationsArray[slotPermutation][2],
                                    rotorPosition1,
                                    rotorPosition2,
                                    rotorPosition3]
    # return iteration_best
    # print(slotPermutation)

num_cores = multiprocessing.cpu_count()

arr = np.array([0, 1, 2, 3, 4])
permutations_list = list(permutations(arr, 3))
slotPermutationsArray = np.array(permutations_list)

if slotPermutationsArray.shape[0] <= num_cores:
    num_threads = slotPermutationsArray.shape[0]
else:
    num_threads = num_cores
    iterations = slotPermutationsArray.shape[0]/num_cores

if slotPermutationsArray.shape[0] < num_cores:
    num_threads = slotPermutationsArray.shape[0]
    num_iterations = 1
    num_extra_iterations = 0
else:
    num_threads = num_cores
    num_iterations = slotPermutationsArray.shape[0] // num_cores
    num_extra_iterations = operator.mod(slotPermutationsArray.shape[0],num_cores)

results = np.array([])
threads = []
for iteration in range(num_iterations):
    for thread in range(num_threads):
        t = threading.Thread(target=enigmaTest(operator.add(thread,(operator.mul(iteration,num_threads)))))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

for thread in range(num_extra_iterations):
    t = threading.Thread(target=enigmaTest(operator.add(thread,(operator.mul(iteration,num_threads)))))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

end_time = time.time()
print(operator.sub(end_time,start_time))
print(results)