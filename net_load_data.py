from numpy import *
import numpy as np
import cPickle
def load_data():
    """载入解压后的数据，并读取"""
    with open('data/mnist_pkl/mnist.pkl','rb') as f:
        try:
            train_data,validation_data,test_data = cPickle.load(f)
            print " the file open sucessfully"
            # print train_data[0].shape  #(50000,784)
            # print train_data[1].shape   #(50000,)
            return (train_data,validation_data,test_data)
        except EOFError:
            print 'the file open error'
            return None
 
def data_transform():
    """将数据转化为计算格式"""
    t_d,va_d,te_d = load_data()
    # print t_d[0].shape  # (50000,784)
    # print te_d[0].shape  # (10000,784)
    # print va_d[0].shape  # (10000,784)
    # n1 = [np.reshape(x,784,1) for x in t_d[0]] # 将5万个数据分别逐个取出化成（784,1），逐个排列
    n = [np.reshape(x, (784, 1)) for x in t_d[0]]  # 将5万个数据分别逐个取出化成（784,1），逐个排列
    # print 'n1',n1[0].shape
    # print 'n',n[0].shape
    m = [vectors(y) for y in t_d[1]] # 将5万标签（50000,1）化为（10,50000）
    train_data = zip(n,m)  # 将数据与标签打包成元组形式
    n = [np.reshape(x, (784, 1)) for x in va_d[0]]  # 将5万个数据分别逐个取出化成（784,1），排列
    validation_data = zip(n,va_d[1])   # 没有将标签数据矢量化
    n = [np.reshape(x, (784, 1)) for x in te_d[0]]  # 将5万个数据分别逐个取出化成（784,1），排列
    test_data = zip(n, te_d[1])  # 没有将标签数据矢量化
    # print train_data[0][0].shape  #(784,）
    # print "len(train_data[0])",len(train_data[0]) #2
    # print "len(train_data[100])",len(train_data[100]) #2
    # print "len(train_data[0][0])", len(train_data[0][0]) #784
    # print "train_data[0][0].shape", train_data[0][0].shape #（784,1）
    # print "len(train_data)", len(train_data)  #50000
    # print train_data[0][1].shape  #(10,1)
    # print test_data[0][1] # 7
    return (train_data,validation_data,test_data)
def vectors(y):
    """赋予标签"""
    label = np.zeros((10,1))
    label[y] = 1.0 #浮点计算
    return label
