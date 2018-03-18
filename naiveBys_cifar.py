#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np

# 训练集文件
## 读取未处理的数据
images_file = './data/cifar-10-batches-py/data_batch_'

def load_CIFAR_batch(filename):
  """load single batch of cifar """
  with open(filename, 'rb') as f:
      datadict = pickle.load(f, encoding='latin1')
      X = datadict['data']
      Y = datadict['labels']
      X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
      X = X.reshape(10000, 3 * 32 * 32)
      Y = np.array(Y)
      return X, Y


def naiveBysTrain(train, label):
    """
    朴素贝叶斯分类对样本的训练
    :param train: 所有样本集
    :param label: 所有样本标签
    :return: 训练得到的参数
    """
    classTrain = {} #定义 dictionary
    classParam1 = {}
    classParam2 = {}
    image_num = train.shape[0]
    for i in range(image_num):
        label_num = label[i]
        classTrain[label_num] = np.row_stack((classTrain.get(label_num, train[i]), train[i]))
    for key, value in classTrain.items():
        value = value[1:value.shape[0]] #删去第一行（由循环引入）
        classParam1[key] = np.mean(value, axis = 0, dtype = np.float32) #每个像素均值

        #测试用
        #plt.imshow(np.reshape(classParam1[key], (28, 28))*225)
        #plt.show()

        classParam2[key] = np.var(value, axis = 0, dtype = np.float32) #每个像素方差
    return classParam1, classParam2


def naiveBysClassify(test, train_param1, train_param2):
    """
    朴素贝叶斯分类
    :param test: 一个测试数据
    :param train_param: 训练得到的参数（每类的像素点的概率和每类的先验概率）
    :return: 预测结果
    """
    # 计算概率并投出最大可能
    maxVote = 0
    maxProb = 0

    for key, value in train_param1.items():
        prob = 1
        exp = -(test - value) ** 2 / train_param2[key]
        prob_i = (2 * np.pi * train_param2[key])**(-0.5)*np.exp(exp) * 1000000 + 1
        #print(prob_i)
        for i in range(len(prob_i)):
            prob *= int(prob_i[i]) #累乘得到的数组

        if prob > maxProb:
            maxProb = prob
            maxVote = key

    return maxVote


def run():
    """
    主函数
    :return: 无
    """
    #step one: load
    print('loading...')

    # 未处理的像素点
    # train_image,train_label = load_CIFAR_batch(images_file + '1')
    # test_image, test_label = load_CIFAR_batch(images_file + '3')
    # test_image = test_image[:2000]
    # test_label = test_label[:2000]

    # 处理后的gist特征点
    f = np.loadtxt('./data/Labels_Features.txt')
    labels = f[:, 0]
    images = f[:, 1:]

    train_image = images[:40000] * 10
    train_label = labels[:40000]
    test_image = images[40000:] * 10
    test_label = labels[40000:]

    #step two: train
    print('training...')
    TrainStart = time.clock()
    param1, param2 = naiveBysTrain(train_image, train_label)
    TrainEnd = time.clock()
    print('train time %.2f s' % (TrainEnd - TrainStart))

    #step three: test
    print('testing...')

    time1 = time.clock()
    cnt_match = 0
    test_num = test_image.shape[0]
    for i in range(test_num):
        predict = naiveBysClassify(test_image[i], param1, param2)
        if predict == test_label[i]:
            cnt_match += 1

        if (i + 1) % 100 == 0:
            print('%d pictures are finished \r' % (i + 1), end = '')

    accuracy = float(cnt_match / test_num)
    time2 = time.clock()

    # step four: show result
    print('\n test time %.2f s' % (time2 - time1))
    print('Accuracy is %.2f%%' % (accuracy * 100))


if __name__ == '__main__':
    run()
