from PIL import Image
from collections import Counter
import numpy as np
import os
from math import log2
# import matplotlib.pyplot as plt


def count_entropy(matrix, e_min, e_max, size, name):
    occur = Counter(matrix)
    p = np.zeros((e_max,), dtype=float)
    entropy = 0

    #prawdopodobienstwo
    for key in occur.keys():
        p[key] += (abs(occur[key]) / (size[0]*size[1]))

    for i in range(e_min, e_max):   #log2(0) = -inf
        if(p[i] == 0):
            continue

        entropy += p[i]*log2(p[i])  

    # if name == "lennagrey.bmp":
    #     x = np.arange(0,e_max)
    #     plt.plot(x, p)
    #     plt.title(name)
    #     plt.show()

    return -entropy

def differential_coding(matrix):
    rows, cols = matrix.shape
    result = np.zeros_like(matrix)

    result[0, 0] = matrix[0, 0]

    # dla pierwszego wiersza
    for j in range(1, cols):
        result[0, j] = matrix[0, j] - matrix[0, j - 1]

    #dla pierwszej kolumny
    for i in range(1, rows):
        result[i, 0] = matrix[i, 0] - matrix[i - 1, 0]

    #dla reszty macierzy
    for i in range(1, rows):
        for j in range(1, cols):
            result[i, j] = matrix[i, j] - matrix[i, j-1]

    return result


def diff_decode(matrix):
    rows, cols = matrix.shape
    result = np.zeros_like(matrix)

    result[0, 0] = matrix[0, 0]

    # Dla pierwszego wiersza
    for j in range(1, cols):
        result[0, j] = matrix[0, j] + result[0, j - 1]

    # Dla pierwszej kolumny
    for i in range(1, rows):
        result[i, 0] = matrix[i, 0] + result[i - 1, 0]

    # Dla reszty macierzy
    for i in range(1, rows):
        for j in range(1, cols):
            result[i, j] = matrix[i, j] + result[i, j-1]

    return result

# def differential_coding(matrix):
#     rows, cols = matrix.shape
#     result = np.zeros_like(matrix, dtype=int)

#     result[0, 0] = matrix[0, 0]

#     for i in range(rows):
#         for j in range(cols):
#             if i!=0 and j==0:
#                 result[i, j] = matrix[i, j] - matrix[i-1, j]
#             else:
#                 result[i, j] = matrix[i, j] - matrix[i, j-1]


#     return result


# def diff_decode(matrix):
#     rows, cols = matrix.shape
#     result = np.zeros_like(matrix, dtype=int)

#     result[0, 0] = matrix[0, 0]

#     for i in range(rows):
#         for j in range(cols):
#             if i != 0 and j == 0:
#                 result[i, j] = matrix[i, j] + result[i-1, j]
#             else:
#                 result[i, j] = matrix[i, j] + result[i, j-1]


#     return result


dir = "C:/Users/bolec/OneDrive/Pulpit/TIIK/lab1.2/Image2"
img_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
entropies = []
entropies_diff = []

for file in img_files:
    path = os.path.join(dir, file)
    img = Image.open(path)
    
    entropies.append(count_entropy(np.array(img).flatten(), 0, 255, img.size, file))
    diff_encoded_img = differential_coding(np.array(img))
    diff_entropy = count_entropy(diff_encoded_img.flatten(), 0, 511, img.size, file)

    decode_img = diff_decode(diff_encoded_img)
    print(np.array(img), "\n\n\n\n",decode_img)
    if np.array_equal(np.array(img), decode_img):
        print(True)
    else:
        print(False)

    entropies_diff.append(diff_entropy)

    print(f"File: {file}\t\tentropy:  {entropies[len(entropies)-1]}\t\tentropy differ: {entropies_diff[len(entropies_diff)-1]}")

print(f"\nAverage val: {np.mean(entropies)}")