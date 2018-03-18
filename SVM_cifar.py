#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

#线性核函数，对于J的求和被矩阵替代
def kernel(trainMatrix, i):
    return trainMatrix.dot(trainMatrix[i].transpose())

def smoComputation(trainMatrix, labelMatrix, C, toler, maxIter):
    b = 0
    m = trainMatrix.shape[0]#m: mums of examples, n: numb of features
    n = trainMatrix.shape[1]
    alphas = np.zeros(m)#alphas:m*1
    iter = 0
    E = {}
    j_blocked = {}
    i_blocked = {}
    while (iter < maxIter):
        #alphaPairsChanged = 0
        #第1个变量i选择
        violatedMax, I = 0, -1
        for i in range(m):
            G_xi = (alphas * labelMatrix).dot(kernel(trainMatrix, i)) + b
            E[i] = G_xi - labelMatrix[i]
            #违反KKT判断:
            if(alphas[i] == 0 and labelMatrix[i] * G_xi < 1-toler):
                errAbsolu =  abs(1-toler - labelMatrix[i] * G_xi)
                if errAbsolu > violatedMax and i not in i_blocked:
                    violatedMax = errAbsolu
                    I = i
            if alphas[i] > 0 and alphas[i] < C and (labelMatrix[i] * G_xi < 1-toler or labelMatrix[i] * G_xi > 1 + toler):
                errAbsolu = max(abs(1 - toler - labelMatrix[i] * G_xi), abs(1 + toler - labelMatrix[i] * G_xi))
                if errAbsolu > violatedMax and i not in i_blocked:
                    violatedMax = errAbsolu
                    I = i
            if alphas[i] == C and labelMatrix[i] * G_xi > 1 + toler:
                errAbsolu =  abs(1 + toler - labelMatrix[i] * G_xi)
                if errAbsolu > violatedMax and i not in i_blocked:
                    violatedMax = errAbsolu
                    I = i

        #第2个变量j选择
        J = -1
        if(I != -1):
            if(E[I] > 0):
                minE2 = max(E.values()) + 1
                for key in E:
                    if(E[key] < minE2 and key != I and key not in j_blocked):
                        minE2 = E[key]
                        J = key
            if(E[I] < 0):
                maxE2 = min(E.values()) - 1
                for key in E:
                    if (E[key] > maxE2 and key != I and key not in j_blocked):
                        maxE2 = E[key]
                        J = key
        if I == -1 and J == -1:
            break
        if I != -1 and J == -1:
            #print("J work for All J in I:%d ---> choose another I" %I)
            j_blocked.clear()
            i_blocked[I] = 1
            continue
        # 第2个变量j选择
        alphaIold = alphas[I].copy()
        alphaJold = alphas[J].copy()
        if (labelMatrix[I] != labelMatrix[J]):
            L = max(0, alphaJold - alphaIold)
            H = min(C, C + alphaJold - alphaIold)
        else:
            L = max(0, alphaJold + alphaIold - C)
            H = min(C, alphaJold + alphaIold)
        if L == H:
            #print("L == H for I:%d, J:%d --> choose another J" %(I, J))
            j_blocked[J] = 1
            continue
        eta = trainMatrix[I].dot(trainMatrix[I].transpose()) + trainMatrix[J].dot(trainMatrix[J].transpose())
        eta += -2.0 * trainMatrix[I].dot(trainMatrix[J].transpose())
        alphas_unCut = float(alphaJold + labelMatrix[J] * (E[I] - E[J]) / eta)

        if (abs(clipAlpha(alphas_unCut, H, L) - alphaJold) < 0.0001):
            #print("J not moving enough for I:%d, J:%d ---> choose another J" % (I, J))
            j_blocked[J] = 1
            continue

        #更新I，J
        alphas[J] = clipAlpha(alphas_unCut, H, L)
        alphas[I] = alphaIold + labelMatrix[J] * labelMatrix[I] * (alphaJold - alphas[J])

        #更新b
        b1 = b - E[I]
        b1 += - labelMatrix[I] * (alphas[I] - alphaIold) * trainMatrix[I].dot(trainMatrix[I].transpose())
        b1 += - labelMatrix[J] * (alphas[J] - alphaJold) * trainMatrix[I].dot(trainMatrix[J].transpose())

        b2 = b - E[J]
        b2 += - labelMatrix[I] * (alphas[I] - alphaIold) * trainMatrix[I].dot(trainMatrix[J].transpose())
        b2 += - labelMatrix[J] * (alphas[J] - alphaJold) * trainMatrix[J].dot(trainMatrix[J].transpose())

        if (0 < alphas[I]) and (C > alphas[I]):
            b = b1
        elif (0 < alphas[J]) and (C > alphas[J]):
            b = b2
        else:
            b = (b1 + b2) / 2.0
        #print("iter: %d"%iter)
        #print("i:%d from %f to %f"%(I, float(alphaIold), alphas[I]))
        #print("j:%d from %f to %f" % (J, float(alphaJold), alphas[J]))
        iter += 1
        j_blocked.clear() #Reset Block list
        i_blocked.clear()
        #print("iteration number: %d" % iter)
    w = (alphas * labelMatrix).dot(trainMatrix)
    return b, w


def binary(x, i):
    if(x == i):
        return 1
    else:
        return -1

def multiTrain(train, label):
    """
    十分类器训练
    :param train: 训练集
    :param label: 训练标签
    :return: 分类器数据
    """

    param_b = {}
    param_w = {}
    image_num = train.shape[0]
    C = 10
    toler = 0.05
    max_generation = 100

    for i in range(10):
        bi_label = list(map(binary, label, np.zeros(image_num) + i))
        param_b[i], param_w[i] = smoComputation(train, np.array(bi_label), C, toler, max_generation)
    return param_b, param_w


def TestSVM(test, b, w):
    """
    对二分类器测试结果
    :param test: 一个测试数据
    :param svm: 二分类器参数
    :return: 二值化的标签
    """
    estimate = np.dot(w, test.transpose()) + b
    #print('estimate:')
    #print(estimate)

    if(estimate > 0):
        label = 1
    else:
        label = -1
    return label


def multiClassify(test, b, w):
    """
    多分类器测试
    :param test: 一个测试集
    :param param: 训练参数（十个分类器）
    :return: 预测标签
    """
    vote = 0
    for key,value in b.items():
        predict = TestSVM(test, value, w[key])
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

    # 处理后的gist特征点
    f = np.loadtxt('./data/Labels_Features.txt')
    labels = f[:, 0]
    images = f[:, 1:]

    train_image = images[:600] * 10
    train_label = labels[:600]
    test_image = images[40001:] * 10
    test_label = labels[40001:]

    # step two: train
    print('training...')
    trainStart = time.clock()
    param_b, param_w = multiTrain(train_image, train_label)
    trainEnd = time.clock()
    print('\n train time %.2f s' % (trainEnd - trainStart))

    # step three: test
    print('testing...')

    TestStart = time.clock()
    test_num = test_image.shape[0]
    cnt_match = 0
    for i in range(test_num):
        predict = multiClassify(test_image[i], param_b, param_w)
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