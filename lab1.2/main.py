from PIL import Image
from collections import Counter
import numpy as np
import os
from math import log2

def count_entropy(matrix, e_min, e_max, size):
    occur = Counter(matrix)
    p = np.zeros((255,), dtype=float)
    entropy = 0

    #prawdopodobienstwo
    for key in occur.keys():
        p[key] += (abs(occur[key]) / (size[0]*size[1]))

    for i in range(e_min, e_max):
        #log2(0) = -inf
        if(p[i] == 0):
            continue

        entropy += p[i]*log2(p[i])  

    return -entropy


dir = "C:/Users/bolec/OneDrive/Pulpit/TIIK/lab1.2/Image2"
img_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
entropies = []

for file in img_files:
    path = os.path.join(dir, file)
    img = Image.open(path)
    
    entropies.append(count_entropy(np.array(img).flatten(), 0, 255, img.size))
    print(f"File: {file}\t\tentropy: ", entropies[len(entropies)-1])


print(f"\nAverage val: {np.mean(entropies)}")