from math import log2
from collections import Counter
import numpy as np
import os
import librosa

def count_entropy(matrix, e_min, e_max, size):
    occur = Counter(matrix)
    p = np.zeros(e_max, dtype=np.float64)
    
    if e_min < 0:
        p1 = np.zeros(-e_min, dtype=np.float64)
    
    entropy = 0

    #prawdopodobienstwo
    for key in occur.keys():
        if key < 0:
            p1[int(key)] += (abs(occur[key]) / (size))
        else:
            p[int(key)] += (abs(occur[key]) / (size))

    if e_min < 0:
        p = np.concatenate((p1, p))
        # print(p.size)

    for i in range(e_min, e_max):   
        if(p[i] == 0):  #log2(0) = -inf
            continue

        entropy += p[i]*log2(p[i])  

    return -entropy


def differ_code(channel):
    channel1 = channel.astype(np.float64)
    result = np.zeros_like(channel1)

    result[0] = channel1[0]

    for i in range(1, len(channel1)):
        result[i] = channel1[i] - channel1[i-1]

    return result


def decode(channel):
    channel1 = channel.astype(np.float64)
    result = np.zeros_like(channel1)

    result[0] = channel1[0]

    for i in range(1, len(channel1)):
        result[i] = channel1[i] + result[i-1]

    return result    


left_entropy = []
right_entropy = []
diff_left_ent = []
diff_right_ent = []
dir = "C:/Users/rubin/OneDrive/Pulpit/TIIK/lab3.4/audio"
audio_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
for file in audio_files:
    audio_data = librosa.load(os.path.join(dir, file), sr=None, mono=False)

    left_channel = audio_data[0][0]
    right_channel = audio_data[0][1]
    left_channel = np.floor(left_channel * float(2**15) + 0.5)
    right_channel = np.floor(right_channel * float(2**15) + 0.5)  

    print(f"\nAudio data {file}:\nleft channel:\n{left_channel}\nright channel:\n{right_channel}\n\n")

    l_ent = count_entropy(left_channel, -32768, 32767, len(left_channel))
    r_ent = count_entropy(right_channel, -32768, 32767, len(right_channel))
    m = (l_ent + r_ent)/2
    print(f"Both mean for {file}: {m}\n")

    left_entropy.append(round(l_ent, 4))
    right_entropy.append(round(r_ent, 4))

    #differ
    left_encoded = differ_code(left_channel)
    right_encoded = differ_code(right_channel)

    l_ent = count_entropy(left_encoded, -65536, 65535, len(left_channel))
    r_ent = count_entropy(right_encoded, -65536, 65535, len(right_channel))
    m = (l_ent + r_ent)/2
    print(f"Both mean entropy' for {file}: {m}")

    diff_left_ent.append(round(l_ent, 4))
    diff_right_ent.append(round(r_ent, 4))

    l_decoded = decode(left_encoded)
    r_decoded = decode(right_encoded)

    print(f"\nLEFT\nORIGINAL: {left_channel}\nDECODED: {l_decoded}\n\nRIGHT\nORIGINAL: {right_channel}\nDECODED: {r_decoded}")

    if np.array_equal(left_channel, l_decoded) and np.array_equal(right_channel, r_decoded):
        print("\n\nSAME: TRUE\n-----------------------------------------------------------")
    else:
        print("\n\nSAME: FALSE\n-----------------------------------------------------------")

print(f"NOT DIFFER\n\nleft_entropies:\n{left_entropy}\n\nright_entropies:\n{right_entropy}\n\nDIFFER\n\nleft entropy':\n{diff_left_ent}\n\nright entropy':\n{diff_right_ent}")