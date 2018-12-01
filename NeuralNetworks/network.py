#!/usr/bin/env python
# coding: utf-8




import numpy as np
import Feature_Engineering
import metrics
import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine


class layer():
    
    '''
    层。
    储存W,b,A,Z矩阵。
    提供forward（激活），backward（计算在输入值处的激活函数的导数）接口。
    -----
    params:
    Z : 输入值组成的矩阵。shape=(样本个数，本层节点个数)
    A : 激活值组成的矩阵。shape同Z
    B : 偏置单元
    dZ : 残差用于更新W
    dgZ : 激活函数在z点的导数值
    '''
    
    def __init__(self, method='tanh', nodenum=1, lastnodenum=1, size=1):
        
        self.method = method
        self.nodenum = nodenum
        self.lastnodenum = lastnodenum
        self.size = size
        self.W = (np.vstack([np.random.randn(self.lastnodenum,self.nodenum) / 
                             np.sqrt(self.lastnodenum*self.nodenum),
                             np.zeros((1,self.nodenum))]))
        #这里采用cs231n中提到的对ReLU效果很好的一种方式：除以总数的平方根以使得初始的权重方差为1，并且将偏置项权重合并进去，初始化为零
        self.Z = np.ones((self.size,self.nodenum))
        self.A = np.hstack([self.Z.copy(),np.ones((self.size,1))]) #加入了偏置单元
        self.dZ = self.Z.copy()
        self.dgZ = self.A.copy()
        
    def _ReLU(self, Z, direction):
        if direction=='forward':
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    if Z[i][j]<=0:
                        Z[i][j] = 0
            return Z
        for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    if Z[i][j]<=0:
                        Z[i][j] = 0
                    else: 
                        Z[i][j] = 1
        return Z
    
    def _LeakyReLU(self, Z, direction):
        if direction=='forward':
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    if Z[i][j]<=0:
                        Z[i][j] *= 0.01
            return Z
        for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    if Z[i][j]<=0:
                        Z[i][j] = 0.01
                    else: 
                        Z[i][j] = 1
        return Z
    
    def _maxout(self, Z, direction, l_num=2):
        
        if direction=='forward':
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    if Z[i][j]<=0:
                        Z[i][j] = w1*Z[i][j] + b1
                    else: 
                        Z[i][j] = w2*Z[i][j] + b2
            return Z
        for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    if Z[i][j]<=0:
                        Z[i][j] = w1
                    else: 
                        Z[i][j] = w2
        return Z
    
    def _sigmoid(self, z, direction):
        haha = (1 / (1 + np.exp(-z)))
        if direction=='forward':
            return haha
        return haha*(1-haha)
    
    def _tanh(self, z, direction):
        if direction=='forward':
            return ((np.exp(2*z) - 1) / (np.exp(2*z) +1))
        return (4 * (np.exp(2*z) / ((np.exp(2*z) + 1)**2)))
    
    def _test(self, z, direcction):
        if direction=='forward':
            return z
        return 1
    
    def _softmax(self, Z, direction):
        if direction=='forward':
            for z in Z:
                z -= np.max(z)
                z = z / mp.sum(z)
            return Z
        pass #softmax层不用求本地梯度
    
        
    def forward(self):
        
        if self.method=='ReLU':
            self.A = np.hstack([self._ReLU(self.Z, direction='forward'),np.ones((self.size,1))])
        elif self.method=='sigmoid':
            self.A = np.hstack([self._sigmoid(self.Z, direction='forward'),np.ones((self.size,1))])
        elif self.method=='LeakyReLU':
            self.A = np.hstack([self._LeakyReLU(self.Z, direction='forward'),np.ones((self.size,1))])
        elif self.method=='maxout':
            self.A = np.hstack([self._maxout(self.Z, direction='forward'),np.ones((self.size,1))])
        elif self.method=='tanh':
            self.A = np.hstack([self._tanh(self.Z, direction='forward'),np.ones((self.size,1))])
        elif self.method=='test':
            self.A = np.hstack([self._test(self.Z, direction='forward'),np.ones((self.size,1))])
        return self
    
    
    def backward(self):
        
        if self.method=='ReLU':
            self.dgZ = self._ReLU(self.Z, direction='backward')
        elif self.method=='LeakyReLU':
            self.A = self._LeakyReLU(self.Z, direction='backward')
        elif self.method=='maxout':
            self.A = self._maxout(self.Z, direction='backward')
        elif self.method=='sigmoid':
            self.dgZ = self._sigmoid(self.Z, direction='backward')
        elif self.method=='tanh':
            self.dgZ = self._tanh(self.Z, direction='backward')
        elif self.method=='test':
            self.dgZ = self._test(self.Z, direction='backward')
        return self
    
    

class neuralnetwork():
    
    def __init__(self, X_train, Y_train, layer_num=3, eta=0.5, lamda=0.1, hidlayer_nodenum=10, method='softmax'):
        self.method = method
        self.X_train = X_train
        self.Y_train = Y_train
        self.layer_num = layer_num
        self.hidlayer_nodenum = hidlayer_nodenum
        self.eta = eta
        self.lamda = lamda
        #构建网络
        self.size = self.X_train.shape[0]
        self.layers = []
        #输入层：
        inlayer = layer(nodenum=X_train.shape[1], size=self.size)
        self.layers.append(inlayer)
        self.__iter = 0
        #隐藏层和输出层：
        if self.layer_num>=3:
            hidlayer = layer(lastnodenum=self.X_train.shape[1], nodenum=self.hidlayer_nodenum, size=self.size)
            self.layers.append(hidlayer)
            if self.layer_num>3:
                for i in range(self.layer_num-3):                    
                    hidlayer = layer(nodenum=self.hidlayer_nodenum, lastnodenum=self.hidlayer_nodenum, size=self.size)
                    self.layers.append(hidlayer)
            outlayer = layer(nodenum=self.Y_train.shape[1], lastnodenum=self.hidlayer_nodenum, 
                            size=self.size, method='sigmoid')
            self.layers.append(outlayer)
        else: #即layernum=2时
            outlayer = layer(nodenum=self.Y_train.shape[1], lastnodenum=self.X_train.shape[1], 
                             size=self.size, method='sigmoid')
            self.layers.append(outlayer)
        #网络构建完毕。
    
    def forward(self):
        
        #对第一层：
        self.layers[0].A = np.hstack((self.X_train.copy(),np.ones((self.size,1))))
                                     
        
        #接下来的每一层：
        for i in range(1, self.layer_num):
            self.layers[i].Z = self.layers[i-1].A.dot(self.layers[i].W)
            self.layers[i].forward()
            
        return self
    
    def backward(self):
        
        # 从输出层开始：
        if self.method=='sigmoid':
            self.layers[-1].backward()
            self.layers[-1].dZ = ((self.layers[-1].A[:,:(self.layers[-1].A.shape[1]-1)] - self.Y_train) * 
                                  (self.layers[-1].dgZ))
        if self.method=='softmax':
            for i in range(self.layers[-1].dZ.shape[0]):
                self.layers[-1].dZ[i][self.Y_train[i]==1] = (self.layers[-1].A[i,:(self.layers[-1].A.shape[1]-1)])[self.Y_train[i]==1] - 1
                self.layers[-1].dZ[i][self.Y_train[i]==0] = (self.layers[-1].A[i,:(self.layers[-1].A.shape[1]-1)])[self.Y_train[i]==0]
        
        # 依次往后传播,计算dz：
        num = self.layer_num #为了下面的代码简洁一点
        for i in range(num-1):
            self.layers[num-2-i].backward()
            self.layers[num-2-i].dZ=(self.layers[num-1-i].dZ.dot(self.layers[num-1-i].W[:(self.layers[num-1-i].W.shape[0]-1),:].T)
                                     * self.layers[num-2-i].dgZ)
            
        # 更新W,b：
        
        for i in range(1,num):
            dW = (self.layers[i].W * self.lamda) + ((self.layers[i-1].A.T.dot(self.layers[i].dZ)) / self.size)
            self.layers[i].W = self.layers[i].W - (self.eta * dW)
            
        self.__iter += 1
        
        return self
    
    
class NeuralNetworkClassifier():
    
    def __init__(self, network, X_train, Y_train, eta=0.005, lamda=0.01, epsilon=1e-4, layer_num=3, hidlayer_nodenum=10, method='softmax'):
        self.network = network
        self.X_train = X_train
        self.Y_train = Y_train
        self.layer_num = layer_num
        self.hidlayer_nodenum = hidlayer_nodenum
        self.method = method
        self.eta = eta
        self.epsilon = epsilon
        self.lamda = lamda
        self.mynetwork = neuralnetwork(self.X_train, self.Y_train, layer_num=layer_num, lamda=self.lamda)
        
    def _J(self, network, Y_train):
        y_pred = network.layers[-1].A[:,:(network.layers[-1].A.shape[1]-1)]
        W2 = 0
        J = 0
        for i in range(network.layer_num):
            W2 += np.sum(network.layers[i].W ** 2)
        if self.method=='sigmoid':    
            J = 0.5 * np.sum((Y_train - y_pred)**2) + self.lamda * 0.5 * W2
        if self.method=='softmax':
            for i in range(y_pred.shape[0]):
                assert y_pred[i].size >0
                J += -np.log((y_pred[i][self.Y_train[i]==1] / np.sum(y_pred[i])) + 1e-7)
        return J
        
    def fit(self, show='Ture'):
        self.mynetwork.forward()
        _iter = 0
        J_history = []
        while(_iter<1000):
            J1 = self._J(Y_train=self.Y_train, network=self.mynetwork)
            self.mynetwork.backward()
            self.mynetwork.forward()
            J2 = self._J(Y_train=self.Y_train, network=self.mynetwork)
            delta = J1 -J2
            
            if (delta) < self.epsilon:
                break
            
            _iter += 1
            J_history.append(J2)
        if show=='Ture':
            plt.plot([i for i in range(_iter)],J_history)
            plt.show()
        
        
    def predict(self, X_test, Y_test, layer_num=None):
        if layer_num==None:
            layer_num = self.layer_num
        self.newnetwork = neuralnetwork(X_train=X_test, Y_train=Y_test, layer_num=layer_num)
        for i in range(self.newnetwork.layer_num):
            self.newnetwork.layers[i].W = self.mynetwork.layers[i].W.copy()
        self.newnetwork.forward()
        y_pred = np.array(self.newnetwork.layers[-1].A[:,:(self.newnetwork.layers[-1].A.shape[1]-1)] > 0.5, dtype='int')
        return y_pred
    
    def score(self, X_test, Y_test, scoring=metrics.acc_score):
        y_pred = self.predict(X_test,Y_test).reshape(-1,).astype('float64')
        y_test = Y_test.reshape(-1,)
        return scoring(y_pred, y_test)