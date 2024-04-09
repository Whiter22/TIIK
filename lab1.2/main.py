from PIL import Image
from collections import Counter
import numpy as np
import os
from math import log2
import matplotlib.pyplot as plt


def count_entropy(matrix, e_min, e_max, size, name):
    occur = Counter(matrix)
    p = np.zeros(e_max, dtype=np.float64)
    
    if e_min < 0:
        p1 = np.zeros(-e_min, dtype=np.float64)
    
    entropy = 0

    #prawdopodobienstwo
    for key in occur.keys():
        if key < 0:
            p1[int(key)] += (abs(occur[key]) / (size[0]*size[1]))
        else:
            p[int(key)] += (abs(occur[key]) / (size[0]*size[1]))

    if e_min < 0:
        p = np.concatenate((p1, p))
        # print(p.size)

    for i in range(e_min, e_max):   
        if(p[i] == 0):  #log2(0) = -inf
            continue

        entropy += p[i]*log2(p[i])  

    if name == "lennagrey.bmp":
        x = np.arange(e_min,e_max)
        plt.plot(x, p)
        plt.title(name)
        plt.grid()
        # plt.show()

    return -entropy


def differ_matrix(matrix):
    matrix1 = matrix.astype(np.float64)

    result = np.zeros_like(matrix1)

    result[0, 0] = matrix1[0, 0]
    rows, cols = result.shape

    for i in range(rows):
        for j in range(cols):
            if i != 0 and j == 0:
                result[i, j] = matrix1[i, j] - matrix1[i-1, j]
            else:
                result[i, j] = matrix1[i, j] - matrix1[i, j-1]

    return result


def decode(matrix):
    matrix1 = matrix.astype(np.float64)

    result = np.zeros_like(matrix1)

    result[0, 0] = matrix1[0, 0]
    rows, cols = result.shape

    for i in range(rows):
        for j in range(cols):
            if i != 0 and j == 0:
                result[i, j] = matrix1[i, j] + result[i-1, j]
            else:
                result[i, j] = matrix1[i, j] + result[i, j-1]

    return result


if __name__ == "__main__":
    dir = "C:/Users/bolec/OneDrive/Pulpit/TIIK/lab1.2/Image2"
    img_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    entropies = []
    entropies_diff = []

    for file in img_files:
        path = os.path.join(dir, file)
        img = Image.open(path)
        
        entropies.append(count_entropy(np.array(img).flatten(), 0, 255, img.size, file))
        encoded_img = differ_matrix(np.array(img))
        entropies_diff.append(count_entropy(encoded_img.flatten(), -255, 255, img.size, file))
        # decoded_img = decode(encoded_img)

        if file == 'lennagrey.bmp':
            plt.show()

        print(f"{file} size: {img.size}")
        # print(f"File: {file}\t\tentropy:  {round(entropies[len(entropies)-1], 4)}\t\t entropy': {round(entropies_diff[len(entropies_diff)-1], 4)}")
        # m = np.array(img).astype(np.float64)
        # if np.array_equal(m, decoded_img):
        #     print(True)
        # else:
        #     print(False)

    print(f"\nAverage val entropy: {np.mean(entropies)}\nAverage val entropy': {np.mean(entropies_diff)}") 