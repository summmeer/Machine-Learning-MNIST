#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pickle
import numpy as np

# 训练集文件
## 读取未经处理的数据
images_file = './data/cifar-10-batches-py/data_batch_'

def load_CIFAR_batch(filename):
  """load single batch of cifar """
  with open(filename, 'rb') as f:
      datadict = pickle.load(f, encoding='latin1')
      X = datadict['data']
      Y = datadict['labels']
      X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
      X = X.reshape(10000, 3 * 32 * 32)
      Y = np.array(Y)
      return X, Y


def knnClassify(test, train, labels, k):
    """
    Knn分类器
    :param test: 一个测试集数据
    :param train: 全体训练集数据
    :param labels: 全体训练集标签
    :param k: 近邻比较的个数
    :return: 一个预测的类别标签
    """

    #求距离
    train_num = train.shape[0]
    diff = np.tile(test, (train_num, 1)) - train #和每个训练集数据的距离
    sqr_diff = diff ** 2
    sqr_dis = np.sum(sqr_diff, axis = 1) #每行求和
    dis = sqr_dis ** 0.5
    sort_diss = np.argsort(dis) #从小到大排序并返回索引

    #投出最佳分类
    classCount = {}  #定义一个dictionary
    for i in range(k):
        voteLabel = labels[sort_diss[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxVote = 0
    maxLabel = 0
    for key, value in classCount.items():
        if value > maxVote:
            maxVote = value
            maxLabel = key

    return maxLabel


def run():
    """
    主函数
    :return:无
    """
    #step one: load
    print('loading data...')

    #未处理的像素点
    #train_image,train_label = load_CIFAR_batch(images_file + '1')
    #test_image, test_label = load_CIFAR_batch(images_file + '3')
    #test_image = test_image[:2000]
    #test_label = test_label[:2000]

    #处理后的gist特征点
    f = np.loadtxt('./data/Labels_Features.txt')
    labels = f[:,0]
    images = f[:,1:]

    train_image = images[:40000]
    train_label = labels[:40000]
    test_image = images[40000:]
    test_label = labels[40000:]

    #step two: train
    print('training...')
    pass

    #step three: test
    print('testing...')

    time1 = time.clock()
    test_num = test_image.shape[0]
    cnt_match = 0
    for i in range(test_num):
        predict = knnClassify(test_image[i], train_image, train_label, 3)
        if predict == test_label[i]:
            cnt_match += 1

        if (i + 1) % 100 == 0:
            print('%d pictures are finished \r' % (i + 1), end = '')

    accuracy = float(cnt_match / test_num)
    time2 = time.clock()

    #step four: show result
    print('\n run time %.2f s' %(time2 - time1))
    print('Accuracy is %.2f%%' %(accuracy * 100))


if __name__ == '__main__':
    run()