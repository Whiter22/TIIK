import os
import numpy as np

'''
Przykładowy wynik dla pliku (Lab. 5/6): 915
p = 0.915
k = 3
Długość zakodowana = 4222 bity.
'''

f_k = lambda p: np.ceil(np.log2(np.log2(((np.sqrt(5) - 1)/2))/np.log2(p)))
f_u = lambda n, k: n//(2**k)
f_v = lambda n, k, u: n - (2**k)*u


def rice_enc(N, k):
    u = []
    u = [f_u(n, k) for n in N]

    if k != 0:
        v = []
        v = [f_v(ni, k, ui) for ni, ui in zip(N, u)]

        for i in range(len(v)):
            v[i] = str(bin(v[i]))[2:]
            if len(v[i]) == 1:
                v[i] = ('0' * 2) + v[i]

            elif len(v[i]) == 2:
                v[i] = ('0') + v[i]
    
    if k == 0:
        v = [('0' * 2) for _ in range(len(u))]

    for i in range(len(u)):
        u[i] = '0' * (u[i])
        u[i] += '1'

    out = []
    for ui, vi in zip(u, v):
        out.append(ui + vi)
        # out.append(ui + ':' + vi)
        # print(out[-1])

    return out


dir = 'C:/Users/bolec/OneDrive/Pulpit/TIIK/lab5.6/streams'
streams = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
OUT = []

for stream in streams:
    with open(os.path.join(dir, stream), 'r') as file:
        content = file.read()
        one_count = content.count('1')

    print(f'file name: {stream}')
    p1 = round(one_count/len(content), 3)
    p0 = 1 - p1
    print(f'p(0): {p0}')
    k = int(f_k(p0))
    print(f'k: {k}')

    counter = 0
    N = []
    u = []
    v = []
    for n in content:
        counter+=1

        if n == '1':
            N.append(counter-1)
            counter = 0
    
    OUT = rice_enc(N, k)

    with open(f'{dir}/{stream}_OUT', 'w+') as file:
        for stream in OUT:
            file.write(stream)

        file.seek(0)
        print(f'Dlugosc zakodowania = {len(file.read())} bity')
    # np.savetxt(f'{dir}/{stream}_OUT', OUT, fmt='%s')
    