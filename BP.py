#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import struct
import random
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

#一个神经网络
#size:每层神经网络的神经元个数
class Network(object):

    def __init__(self, sizes):
        self.layerNum = len(sizes)
        self.sizes = sizes
        #self.biases = [np.random.randn(y, 1) / y for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.uniform(-1.0, 1.0, (y, 1)) / y for y in sizes[1:]]
        self.weights = [np.random.uniform(-1.0, 1.0, (y, x)) * np.sqrt(6.0 / (x + y)) for x, y in zip(sizes[:-1], sizes[1:])]


    def gety_k(self, x_k):
        """计算输入的当前输出"""
        # reshape
        x_k = np.expand_dims(np.transpose(x_k),axis=1)
        for b, w in zip(self.biases, self.weights):
            x_k = sigmoid(np.dot(w, x_k) + b)
        return x_k

    def vectorizedLabel(self, j):
        vlabel = np.zeros((10, 1))
        vlabel[j] = 1.0
        return vlabel

    def backPropagation(self, x, y):
        """向后计算梯度"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 向前计算
        activation = np.expand_dims(np.transpose(x),axis=1)
        activations = [activation]  # 每层的经过激活函数后的值
        layerValue = []  # 每层的输入值
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            layerValue.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 向后计算梯度
        vlabel = self.vectorizedLabel(int(y))
        delta = (activations[-1] - vlabel) * sigmoid_prime(layerValue[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.layerNum):
            z = layerValue[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)


    def update_mini_batch(self, mini_batch, eta):
        """
        更新weights和biases
        :param mini_batch: 小部分训练样本
        :param eta: 学习率
        :return: 无
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backPropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]


    def TestNetwork(self, test_data, test_label):
        """统计准确度"""
        cnt_match = 0

        for i in range(test_label.shape[0]):
            predict = np.argmax(self.gety_k(test_data[i]))
            if predict == int(test_label[i]):
                cnt_match += 1
        return cnt_match


    def EvalNetwork(self, train_data, train_label, epochs, mini_batch_size, eta, test_data, test_label):
        """
        随机梯度下降法进行训练&测试
        :param training_data: 训练数据
        :param epochs: 迭代次数
        :param mini_batch_size: batch长度
        :param eta: 学习率
        :param test_data: 测试数据
        :return: 无
        """
        test_num = test_data.shape[0]
        testResult = []
        train_data_label = list(zip(train_data, train_label))
        train_num = len(train_data_label)

        for j in range(epochs):

            random.shuffle(train_data_label)
            mini_batches = [train_data_label[k:k + mini_batch_size] for k in range(0, train_num, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            EachResult = self.TestNetwork(test_data, test_label)
            testResult.append(EachResult / test_num)
            print("Epoch {0}: {1} / {2}".format(j, EachResult, test_num))
            #eta = eta * 0.95

        pl.plot(range(epochs), testResult)
        pl.show()


#激活函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """求激活函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))

def run():
    """
    主函数
    :return: 无
    """
    print('loading data...')

    train_image = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_label = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_image = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_label = decode_idx1_ubyte(test_labels_idx1_ubyte_file)

    print('evaluating network...')
    StartTime = time.clock()
    net = Network([train_image.shape[1], 50, 10])
    #train_data, train_label, epochs, mini_batch_size, eta, test_data, test_label
    net.EvalNetwork(train_image, train_label, 60, 1000, 0.5, test_image, test_label)
    EndTime = time.clock()

    print('Total time %.2f s' % (EndTime - StartTime))


if __name__ == '__main__':
    run()