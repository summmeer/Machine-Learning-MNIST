#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import struct
from PIL import Image
import matplotlib.pyplot as plt

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
    images[images <= 112] = 0 # 矩阵二值化
    images[images > 112] = 1
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
        classParam1[key] = np.sum(value, axis = 0, dtype = np.float32) / value.shape[0] #每个像素为1的概率

        #测试用
        #plt.imshow(np.reshape(classParam1[key], (28, 28))*225)
        #plt.show()

        classParam2[key] = int(np.float32(value.shape[0] / image_num) * 100000 + 1)
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
        prob_i = (test * value + (-1 * test + 1) * (1 - value)) * 100000 + 1
        plt.imshow(np.reshape(prob_i, (28, 28)) * 225)
        plt.show()
        for i in range(len(prob_i)):
            prob *= int(prob_i[i]) #累乘得到的数组

        prob *= train_param2[key] #乘上先验概率

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

    train_image = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_label = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    # 截取部分
    train_image = train_image[:500]
    train_label = train_label[:500]
    test_image = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_label = decode_idx1_ubyte(test_labels_idx1_ubyte_file)

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
