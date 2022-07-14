# https://blog.csdn.net/SGDBS233/article/details/125785527

import numpy as np
import random
import os


class SVD:
    def __init__(self, mat, K=20):
        self.mat = np.array(mat)
        self.K = K
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.mat[:, 2]) # 电影平均分
        for i in range(self.mat.shape[0]): # mat.shape[0],表示n*m矩阵的n
            uid = self.mat[i, 0]
            iid = self.mat[i, 1]
            self.bi.setdefault(iid, 0) # 向字典中插入
            self.bu.setdefault(uid, 0)
            self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K)) #随机给每个人，项目赋随机因子
            self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))

    def predict(self, uid, iid):  # 预测评分的函数
        # setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu，并设置初始值为0
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.K, 1)))
        self.pu.setdefault(uid, np.zeros((self.K, 1)))
        rating = self.avg + self.bi[iid] + self.bu[uid] + np.sum(self.qi[iid] * self.pu[uid])  # 预测评分公式
        # 由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    def train(self, steps=30, gamma=0.04, Lambda=0.15):  # 训练函数，step为迭代次数。
        print('train data size', self.mat.shape)
        for step in range(steps):
            print('step', step + 1, 'is running')
            KK = np.random.permutation(self.mat.shape[0])  # 随机梯度下降算法，kk为对矩阵进行随机洗牌
            rmse = 0.0
            mae = 0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                eui = rating - self.predict(uid, iid)
                rmse += eui ** 2
                mae += abs(eui)
                self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
                self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
                tmp = self.qi[iid]
                self.qi[iid] += gamma * (eui * self.pu[uid] - Lambda * self.qi[iid])
                self.pu[uid] += gamma * (eui * tmp - Lambda * self.pu[uid])
            gamma = 0.93 * gamma  # gamma以0.93的学习率递减
            print('rmse is {0:3f}, ase is {1:3f}'.format(np.sqrt(rmse / self.mat.shape[0]), mae / self.mat.shape[0]))

    def test(self, test_data):

        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0
        mae = 0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating - self.predict(uid, iid)
            rmse += eui ** 2
            mae += abs(eui)
        print('rmse is {0:3f}, ase is {1:3f}'.format(np.sqrt(rmse / self.mat.shape[0]), mae / self.mat.shape[0]))


def getData(file_name):
    """
    获取训练集和测试集的函数
    """
    data = []
    with open(os.path.expanduser(file_name), encoding='utf-8') as f:
        for line in f.readlines():
            list = line.split()
            data.append([int(i) for i in list[:3]])
    random.shuffle(data)
    train_data = data[:int(len(data) * 7 / 10)]
    test_data = data[int(len(data) * 7 / 10):]
    print('load data finished')
    print('total data ', len(data))
    return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = getData('./u.data')
    a = SVD(train_data, 30)
    a.train()
    a.test(test_data)