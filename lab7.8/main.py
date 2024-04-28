import os
import librosa
import numpy as np
from collections import Counter
from math import log2


f_k = lambda p: np.ceil(np.log2(np.log2(((np.sqrt(5) - 1)/2))/np.log2(p)))
f_u = lambda n, k: n//(2**k)
f_v = lambda n, k, u: n - (2**k)*u
# f_S = lambda e: sum(e)/len(e)

def prob(channel):
    S = np.mean(channel)

    if S >= 2:
        p = (S-1)/S
    else:
        p = 0.5

    return p


def rice_enc(N, k):
    v = []
    u = []
    u = [int(round(f_u(n, k))) for n in N]


    if k != 0:
        v = [int(round(f_v(ni, k, ui))) for ni, ui in zip(N, u)]
        cnt = 0


        for i in range(len(v)):
            vn = ''
            while(v[i] != 0 or cnt != k):
                vn = str(v[i] & 1) + vn
                v[i] >>= 1
                cnt += 1  
            
            cnt = 0
            v[i] = vn
    
    elif k == 0:
        v = ['' for _ in range(len(u))]

    for i in range(len(u)):
        u[i] = '0' * (u[i])
        u[i] += '1'

    out = ''
    for ui, vi in zip(u, v):
        out += (ui + vi)
        # out.append(ui + vi)
        # out.append(ui + ':' + vi)
        # print(out[-1])

    return out


def decode(uv_in, k):
    # ans = ''
    ans = []
    u = 0
    i = 0
    while i < len(uv_in):
        if uv_in[i] != '1':
            u += 1
            i += 1
        
        else:
            v = int(uv_in[i+1: i+k+1], 2)
            i += k+1
            n = u * 2**k + v
            ans.append(n)
            # ans += '0' * n
            # ans += '1'
            u = 0
    
    # print(ans) 


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

        entropy += p[i]*log2(p[i])  

    return -entropy


dir = "C:/Users/rubin/OneDrive/Pulpit/TIIK/lab7.8/audio"
audio_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
for file in audio_files:
    audio_data = librosa.load(os.path.join(dir, file), sr=None, mono=False)
    left_channel = audio_data[0][0]
    right_channel = audio_data[0][1]
    left_channel = np.floor(left_channel * float(2**15) + 0.5)
    right_channel = np.floor(right_channel * float(2**15) + 0.5) 

    #differ
    left_encoded = differ_code(left_channel)
    right_encoded = differ_code(right_channel)

    l_ent = count_entropy(left_encoded, -65536, 65535, len(left_channel))
    r_ent = count_entropy(right_encoded, -65536, 65535, len(right_channel))
    H_m = (l_ent + r_ent)/2
    # print(left_encoded)
    # print(right_encoded)

    for i in range(len(left_encoded)):
        if left_encoded[i] >= 0:
            left_encoded[i] = 2 * left_encoded[i]
        else:
            left_encoded[i] = (-2 * left_encoded[i]) - 1

    p_l = prob(left_encoded)
    k_l = f_k(p_l)

    if k_l > 15:
        k_l = 15

    out_l = rice_enc(left_encoded, k_l)

    for i in range(len(right_encoded)):
        if right_encoded[i] >= 0:
            right_encoded[i] = right_encoded[i] * 2
        else:
            right_encoded[i] = (-2 * right_encoded[i]) - 1

    p_r = prob(right_encoded)
    k_r = f_k(p_r)

    if k_r > 15:
        k_r = 15

    out_r = rice_enc(right_encoded, k_r)

    # # print(out_l)
    # print(len(out_l))
    # print(len(out_r))
    print(file)
    print(f'k_l: {k_l}\nk_r: {k_r}')
    l = len(out_l) + len(out_r)
    # print('coded len: ', l)
    LR = l/(len(right_encoded)+len(left_encoded))
    print('LR: ', LR)

    # l_ent = count_entropy(left_encoded, -65536, 65535, len(left_channel))
    # r_ent = count_entropy(right_encoded, -65536, 65535, len(right_channel))
    # H_m = (l_ent + r_ent)/2

    E = (H_m/LR) * 100
    print(f'Efektywnosc: {E}')  

    # print(out_l)
    # print(round(k_l))
    # print(left_encoded)
    # decode(out_l, round(k_l))