import os
import librosa
import numpy as np
from collections import Counter


f_k = lambda p: np.ceil(np.log2(np.log2(((np.sqrt(5) - 1)/2))/np.log2(p)))
f_u = lambda n, k: n//(2**k)
f_v = lambda n, k, u: n - (2**k)*u
# f_S = lambda e: sum(e)/len(e)

def prob(channel):
    S = np.mean(channel)

    if S >= 2:
        p = (S-1)/2
    else:
        p = 0.5

    return p

def rice_enc(N, k):
    v = []
    u = []
    u = [f_u(n, k) for n in N]

    if k != 0:
        v = [f_v(ni, k, ui) for ni, ui in zip(N, u)]
        cnt = 0

        print(v)

        for i in range(len(v)):
            vn = ''
            while(v[i] != 0 or cnt != k):
                vn = str(int(v[i]) & 1) + vn
                v[i] >>= 1
                cnt += 1  
            
            cnt = 0
            v[i] = vn
    
    elif k == 0:
        v = ['' for _ in range(len(u))]

    for i in range(len(u)):
        u[i] = '0' * (u[i])
        u[i] += '1'

    out = []
    for ui, vi in zip(u, v):
        out.append(ui + vi)
        # out.append(ui + ':' + vi)
        # print(out[-1])

    return out


def differ_code(channel):
    channel1 = channel.astype(np.float64)
    result = np.zeros_like(channel1)

    result[0] = channel1[0]

    for i in range(1, len(channel1)):
        result[i] = channel1[i] - channel1[i-1]

    return result


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

        entropy += p[i]*np.log2(p[i])  

    return -entropy


dir = "C:/Users/rubin/OneDrive/Pulpit/TIIK/lab7.8/audio"
audio_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
for file in audio_files:
    audio_data = librosa.load(os.path.join(dir, file), sr=None, mono=False)
    # print(audio_data)
    left_channel = audio_data[0][0]
    right_channel = audio_data[0][1]
    left_channel = np.floor(left_channel * float(2**15) + 0.5)
    right_channel = np.floor(right_channel * float(2**15) + 0.5)  

    #differ
    left_encoded = differ_code(left_channel)
    right_encoded = differ_code(right_channel)

    for i in range(len(left_encoded)):
        if left_encoded[i] > 0:
            left_encoded[i] = left_encoded[i] * 2
        else:
            left_encoded[i] = (-2 * left_encoded[i]) - 1

    p_l = prob(left_encoded)
    k_l = f_k(p_l)
    out_r = rice_enc(left_encoded, k_l)

    if k_l > 15:
        k_l = 15

    for i in range(len(right_encoded)):
        if right_encoded[i] > 0:
            right_encoded[i] = right_encoded[i] * 2
        else:
            right_encoded[i] = (-2 * right_encoded[i]) - 1

    p_r = prob(right_encoded)
    k_r = f_k(p_r)

    if k_r > 15:
        k_r = 15