# https://blog.csdn.net/SGDBS233/article/details/125838226


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pylab

def loadImage(filename):
    image = Image.open(filename)
    return np.array(image)

def svd(A):
    return np.linalg.svd(A)

def reBuildSVD(U, S, V):
    r = len(S)
    return np.dot(U[:,:r] * S, V[:r,:])

def setZero(U, S, V, k): # 把部分数据清除
    r = len(S)
    for i in range(k, r):
        U[:, i] = 0
        S[i] = 0
        V[i, :] = 0
    return U, S, V

def totalVariation(S, k): # 计算剩余的数据的比例
    return np.sum(S[:k]) / np.sum(S)

def imageProcess(img, k):
    img2 = np.zeros_like(img) # 构建相同的0矩阵
    tv = 1.0
    for c in range(img.shape[2]): #shape[0]图片高度，shape[1]图片宽度，shape[2]图片通道数(彩色图片是3，即R,G,B)
        A = img[:, :, c]
        U, S, V = svd(A)
        tv *= totalVariation(S, k)
        U, S, V = setZero(U, S, V, k)
        img2[:, :, c] = reBuildSVD(U, S, V)
    return img2, tv

def Ratio(A, k): #压缩率
    den = A.shape[0] * A.shape[1] * A.shape[2]
    nom = (A.shape[0] * k + k * A.shape[1] + k) * A.shape[2]
    return 1 - nom/den


filname = "./miku.jpg"
miku = loadImage(filname)

plt.figure(figsize=(20, 10))

## 分别放原图，然后从压缩率高往低
#1
plt.subplot(2, 2, 1)
plt.imshow(miku)
plt.title("origin")


# 2
plt.subplot(2, 2, 2)
img, var = imageProcess(miku, 4 ** 2)
ratio = Ratio(miku, 4 ** 1)
plt.imshow(img)
plt.title('{:.2%} / {:.2%}'.format(var, ratio))

# 3
plt.subplot(2, 2, 3)
img, var = imageProcess(miku, 4 ** 3)
ratio = Ratio(miku, 4 ** 3)
plt.imshow(img)
plt.title('{:.2%} / {:.2%}'.format(var, ratio))


# 4
plt.subplot(2, 2, 4)
img, var = imageProcess(miku, 4 ** 4)
ratio = Ratio(miku, 4 ** 4)
plt.imshow(img)
plt.title('{:.2%} / {:.2%}'.format(var, ratio))

pylab.show()