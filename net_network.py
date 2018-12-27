import numpy as np
import random
class Network(object):   #默认为基类?用于继承：print isinstance(network,object)
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # print 'num_layers', self.num_layers
        self.weight = [np.random.randn(a1, a2) for (a1, a2) in zip(sizes[1:], sizes[:-1])] #产生一个个数组
        self.bias = [np.random.randn(a3,1) for a3 in sizes[1:]]
        # print self.weight[0].shape  #(20,10)
 
    def SGD(self,train_data,min_batch_size,epoches,eta,test_data=False):
        """ 1) 打乱样本，将训练数据划分成小批次
            2）计算出反向传播梯度
            3） 获得权重更新"""
        if test_data: n_test = len(test_data)
        n = len(train_data)   #50000
        random.shuffle(train_data)  # 打乱
        min_batches = [train_data[k:k+min_batch_size] for k in xrange(0,n,min_batch_size)] #提取批次数据
        for k in xrange(0,epoches):   #利用更新后的权值继续更新
            random.shuffle(train_data)  # 打乱
            for min_batch in min_batches:  #逐个传入，效率很低
                self.updata_parameter(min_batch,eta)
            if test_data:
                num = self.evaluate(test_data)
                print "the {0}th epoches: {1}/{2}".format(k,num,len(test_data))
            else:
                print 'epoches {0} completed'.format(k)
 
    def forward(self,x):
        """获得各层激活值"""
        for w,b in zip(self.weight,self.bias):
            x = sigmoid(np.dot(w, x)+b)
        return x
 
    def updata_parameter(self,min_batch,eta):
        """1) 反向传播计算每个样本梯度值
           2） 累加每个批次样本的梯度值
           3） 权值更新"""
        ndeltab = [np.zeros(b.shape) for b in self.bias]
        ndeltaw = [np.zeros(w.shape) for w in self.weight]
        for x,y in min_batch:
            deltab,deltaw = self.backprop(x,y)
            ndeltab = [nb +db for nb,db in zip(ndeltab,deltab)]
            ndeltaw = [nw + dw for nw,dw in zip(ndeltaw,deltaw)]
        self.bias = [b - eta * ndb/len(min_batch) for ndb,b in zip(ndeltab,self.bias)]
        self.weight = [w - eta * ndw/len(min_batch) for ndw,w in zip(ndeltaw,self.weight)]
 
 
    def backprop(self,x,y):
        """执行前向计算，再进行反向传播，返回deltaw,deltab"""
        # [w for w in self.weight]
        # print 'len',len(w)
        # print "self.weight",self.weight[0].shape
        # print w[0].shape
        # print w[1].shape
        # print w.shape
        activation = x
        activations = [x]
        zs = []
        # feedforward
        for w, b in zip(self.weight, self.bias):
            # print w.shape,activation.shape,b.shape
            z = np.dot(w, activation) +b
            zs.append(z)   #用于计算f(z)导数
            activation = sigmoid(z)
            # print 'activation',activation.shape
            activations.append(activation)  # 每层的输出结果
        delta = self.top_subtract(activations[-1],y) * dsigmoid(zs[-1]) #最后一层的delta,np.array乘,相同维度乘
        deltaw = [np.zeros(w1.shape) for w1 in self.weight]  #每一次将获得的值作为列表形式赋给deltaw
        deltab = [np.zeros(b1.shape) for b1 in self.bias]
        # print 'deltab[0]',deltab[-1].shape
        deltab[-1] = delta
        deltaw[-1] = np.dot(delta,activations[-2].transpose())
        for k in xrange(2,self.num_layers):
            delta = np.dot(self.weight[-k+1].transpose(),delta) * dsigmoid(zs[-k])
            deltab[-k] = delta
            deltaw[-k] = np.dot(delta,activations[-k-1].transpose())
        return (deltab,deltaw)
 
    def evaluate(self,test_data):
        """评估验证集和测试集的精度,标签直接一个数作为比较"""
        z = [(np.argmax(self.forward(x)),y) for x,y in test_data]
        zs = np.sum(int(a == b) for a,b in z)
        # zk = sum(int(a == b) for a,b in z)
        # print "zs/zk:",zs,zk
        return zs
 
    def top_subtract(self,x,y):
        return (x - y)
 
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
 
def dsigmoid(x):
    z = sigmoid(x)
    return z*(1-z)
