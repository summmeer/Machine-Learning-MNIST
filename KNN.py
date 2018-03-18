#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import struct

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
    train_num = train.shape[0] / 100
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

    train_image = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_label = decode_idx1_ubyte(train_labels_idx1_ubyte_file)

    # 缩小训练样本数
    #train_image = train_image[:1000]
    #train_label = train_label[:1000]
    test_image = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_label = decode_idx1_ubyte(test_labels_idx1_ubyte_file)

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
