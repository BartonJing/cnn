#coding:utf-8
#数据来源 http://yann.lecun.com/exdb/mnist/

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def norm(label):
    label_vec = []

    label_value = label  # python3中直接就是int
    for i in range(10):
        if i == label_value:
            label_vec.append(1)
        else:
            label_vec.append(0)
    return label_vec



def printImage(data):
    ''' 打印 '''
    aa = np.array(data).reshape(28,28)
    for i in range(len(aa)):
        for j in range(len(aa[i])):
            if aa[i][j] > 0:
                print('*', end='')
            else:
                print(' ', end='')		
        print('')								


