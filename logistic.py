#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import struct
import pylab as pl

# 训练集文件
train_images_idx3_ubyte_file = './data/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = './data/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = './data/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = './data/t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file,'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'#'>IIII'是说使用大端法读取4个 unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)

    # 解析数据集
    image_size = num_rows * num_cols
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, image_size))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        offset += struct.calcsize(fmt_image)
    images /= 255.0 # 矩阵归一化
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集，个数
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file,'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


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
    max_generation = 900
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
    pl.plot(range(max_generation), np.log(cost[1:]))
    pl.show()
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

    train_image = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_label = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_image = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_label = decode_idx1_ubyte(test_labels_idx1_ubyte_file)

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
