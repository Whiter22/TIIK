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
    v = []
    u = []
    u = [f_u(n, k) for n in N]

    if k != 0:
        v = [f_v(ni, k, ui) for ni, ui in zip(N, u)]
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

    out = []
    for ui, vi in zip(u, v):
        out.append(ui + vi)
        # out.append(ui + ':' + vi)
        # print(out[-1])

    return out


def rice_dec(uv_in, k):
    ans = ''
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
            ans += '0' * n
            ans += '1'
            u = 0

    # print(ans)
    return ans


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

    LR = p1 * (k + 1/(1-p0**(2**k)))
    E = ((-p0*np.log2(p0) - p1*np.log2(p1))/LR)*100
    print(f'Efektywnosc: {round(E, 4)}')

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
        content1 = file.read()
        # print(content1)
        
        if k != 0:
            dec =  rice_dec(content1, k)
        
            if dec == content:
                print('Same: True')
            else:
                print('Same: False')

        print(f'Dlugosc zakodowania = {len(content1)} bity')