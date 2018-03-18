#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pickle
import numpy as np
import pylab as pl

# 训练集文件
## 读取未处理的像素点
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


def Trainlogistic(train, labels):
    """
    梯度下降法二分类
    :param train: 训练样本
    :param labels: 二值化标签
    :return: 该分类器的参数
    """
    image_num = train.shape[0]
    pixel_num = train.shape[1]
    param = np.zeros(pixel_num + 1, dtype = np.float32)

    learn_rate = 0.0001
    max_generation = 1000
    min_diff = 0.0001
    cost = np.zeros(max_generation + 1)

    train = np.column_stack((train, np.ones(image_num)))
    for i in range(max_generation):
        if i % 100 == 0:
            learn_rate *= 0.8 #随着迭代次数增加调节learning_rate
        exp = np.dot(param, train.transpose())
        gradient = np.dot(np.exp(exp) / (1 + np.exp(exp)) - labels, train)
        cost[i + 1] = np.sum(-labels * exp + np.log(1 + np.exp(exp)))
        param -= learn_rate * gradient
        if abs(cost[i + 1] - cost[i]) < min_diff:
            break
    #调试用
    #pl.plot(range(max_generation), np.log(cost[1:]))
    #pl.show()
    return param


def Testlogistic(test, param):
    """
    对二分类器测试结果
    :param test: 一个测试数据
    :param param: 二分类器参数
    :return: 二值化的标签
    """
    estimate = np.sum(np.append(test, [1]) * param)
    if(estimate > 0):
        label = 1
    else:
        label = 0
    return label

def binary(x, i):
    return int(x == i)

def multiTrain(train, label):
    """
    十分类器训练
    :param train: 训练集
    :param label: 训练标签
    :return: 分类器数据
    """

    param = {}
    image_num = train.shape[0]

    for i in range(10):
        bi_label = list(map(binary, label, np.zeros(image_num) + i))
        param[i] = Trainlogistic(train, np.array(bi_label))
    return param


def multiClassify(test, param):
    """
    多分类器测试
    :param test: 一个测试集
    :param param: 训练参数（十个分类器）
    :return: 预测标签
    """
    vote = 0
    for key,value in param.items():
        predict = Testlogistic(test, value)
        if(predict == 1):
            vote = key
            break
    return vote


def run():
    """
    主函数
    :return:无
    """
    # step one: load
    print('loading data...')

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
    test_image = images[40001:] * 10
    test_label = labels[40001:]

    # step two: train
    print('training...')
    trainStart = time.clock()
    param = multiTrain(train_image, train_label)
    trainEnd = time.clock()
    print('train time %.2f s' % (trainEnd - trainStart))

    # step three: test
    print('testing...')

    TestStart = time.clock()
    test_num = test_image.shape[0]
    cnt_match = 0
    for i in range(test_num):
        predict = multiClassify(test_image[i], param)
        if predict == test_label[i]:
            cnt_match += 1

        if (i + 1) % 100 == 0:
            print('%d pictures are finished \r' % (i + 1), end='')

    accuracy = float(cnt_match / test_num)
    TestEnd = time.clock()

    # step four: show result
    print('\n test time %.2f s' % (TestEnd - TestStart))
    print('Accuracy is %.2f%%' % (accuracy * 100))


if __name__ == '__main__':
    run()
