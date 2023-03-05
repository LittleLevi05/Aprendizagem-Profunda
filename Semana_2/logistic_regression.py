# -*- coding: utf-8 -*-
"""
@author: miguelrocha
"""

import numpy as np
from dataset import Dataset
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, standardized = False, regularization = False, optimize = False, lamda = 1, alpha = 0.01, iters = 10000):
        self.theta = None
        self.regularization = regularization
        self.optimize = optimize
        self.lamda = lamda
        self.alpha = alpha
        self.iters = iters
        self.standardized = standardized
        self.dataset = None
         
    def printCoefs(self):
        print(self.theta)

    def probability(self, instance):
        x = np.empty([self.dataset.X.shape[1]])
        x[0] = 1
        x[1:] = np.array(instance[:self.dataset.X.shape[1]-1])
        if self.standardized:
            if np.all(self.dataset.sigma!= 0):
                x[1:] = (x[1:] - self.dataset.mu) / self.dataset.sigma
            else: x[1:] = (x[1:] - self.dataset.mu)
        return sigmoid ( np.dot(self.theta, x) )

    def predict_sample(self, instance):
        p = self.probability(instance)
        if p >= 0.5: res = 1
        else: res= 0 
        return res
  
    def costFunction(self, theta = None):
        if theta is None: theta = self.theta
        m = self.dataset.X.shape[0]
        p = sigmoid(np.dot(self.dataset.X,theta))
        cost = (-self.dataset.Y * np.log(p) - (1-self.dataset.Y) * np.log(1-p))
        res = np.sum(cost) / m
        return res
        
    def costFunctionReg(self, theta = None, lamda = 1):
        if theta is None: theta= self.theta
        m = self.dataset.X.shape[0]
        p = sigmoid ( np.dot(self.dataset.X, theta) )
        cost = (-self.dataset.Y * np.log(p) - (1-self.dataset.Y) * np.log(1-p) )
        reg = np.dot(theta[1:], theta[1:]) * lamda / (2*m)
        return (np.sum(cost) / m) + reg

    def gradientDescent(self, alpha = 0.01, iters = 10000):
        m = self.dataset.X.shape[0]
        n = self.dataset.X.shape[1]
        self.theta = np.zeros(n)  
        for its in range(iters):
            J = self.costFunction()
            # if its%1000 == 0: print(J)
            delta = self.dataset.X.T.dot(sigmoid(self.dataset.X.dot(self.theta)) - self.dataset.Y)
            self.theta -= (alpha/m * delta)
            
    def gradientDescentReg(self, alpha = 0.01, iters = 10000):
        m = self.dataset.X.shape[0]
        n = self.dataset.X.shape[1]
        self.theta = np.zeros(n)  
        for its in range(iters):
            J = self.costFunctionReg()
            # if its%1000 == 0: print(J)
            delta = self.dataset.X.T.dot(sigmoid(self.dataset.X.dot(self.theta)) - self.dataset.Y)
            self.theta -= (alpha/m * delta)
            
    def train(self, dataset_train):
        self.dataset = dataset_train
        self.dataset.X = np.hstack((np.ones([dataset_train.X.shape[0],1]), dataset_train.X)) 
        self.dataset.Y = dataset_train.Y
        self.theta = np.zeros(self.dataset.X.shape[1])
        
        if self.regularization:
            if self.optimize: 
                self.optim_model_reg(self.lamda)
            else:
                self.gradientDescentReg(self.alpha, self.iters)
        else:
            if self.optimize:
                self.optim_model()
            else:
                self.gradientDescent(self.alpha, self.iters)

    def optim_model(self):
        from scipy import optimize

        n = self.dataset.X.shape[1]
        options = {'full_output': True, 'maxiter': 500}
        initial_theta = np.zeros(n)
        self.theta, _, _, _, _ = optimize.fmin(lambda theta: self.costFunction(theta), initial_theta, **options)
    
    def optim_model_reg(self, lamda):
        from scipy import optimize

        n = self.dataset.X.shape[1]
        initial_theta = np.ones(n)        
        result = optimize.minimize(lambda theta: self.costFunctionReg(theta, lamda), initial_theta, method='BFGS', options={"maxiter":500, "disp":False} )
        self.theta = result.x    

    def mapX(self):
        self.origX = self.dataset.X.copy()
        mapX = mapFeature(self.dataset.X[:,1], self.dataset.X[:,2], 6)
        self.dataset.X = np.hstack((np.ones([self.dataset.X.shape[0],1]), mapX) )
        self.theta = np.zeros(self.dataset.X.shape[1])

    def plotModel(self):
        from numpy import r_
        pos = (self.dataset.Y == 1).nonzero()[:1]
        neg = (self.dataset.Y == 0).nonzero()[:1]
        plt.plot(self.dataset.X[pos, 1].T, self.dataset.X[pos, 2].T, 'k+', markeredgewidth=2, markersize=7)
        plt.plot(self.dataset.X[neg, 1].T, self.dataset.X[neg, 2].T, 'ko', markerfacecolor='r', markersize=7)
        if self.dataset.X.shape[1] <= 3:
            plot_x = r_[self.dataset.X[:,2].min(),  self.dataset.X[:,2].max()]
            plot_y = (-1./self.theta[2]) * (self.theta[1]*plot_x + self.theta[0])
            plt.plot(plot_x, plot_y)
            plt.legend(['class 1', 'class 0', 'Decision Boundary'])
        plt.show()

    def plotModel2(self):
        negatives = self.origX[self.dataset.Y == 0]
        positives = self.origX[self.dataset.Y == 1]
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.xlim([self.origX[:,1].min(), self.origX[:,1].max()])
        plt.ylim([self.origX[:,1].min(), self.origX[:,1].max()])
        plt.scatter( negatives[:,1], negatives[:,2], c='r', marker='o', linewidths=1, s=40, label='y=0' )
        plt.scatter( positives[:,1], positives[:,2], c='k', marker='+', linewidths=2, s=40, label='y=1' )
        plt.legend()

        u = np.linspace( -1, 1.5, 50 )
        v = np.linspace( -1, 1.5, 50 )
        z = np.zeros( (len(u), len(v)) )

        for i in range(0, len(u)): 
            for j in range(0, len(v)):
                x = np.empty([self.X.shape[1]])  
                x[0] = 1
                mapped = mapFeature( np.array([u[i]]), np.array([v[j]]) )
                x[1:] = mapped
                z[i,j] = x.dot( self.theta )
        z = z.transpose()
        u, v = np.meshgrid( u, v )	
        plt.contour( u, v, z, [0.0, 0.001])
        plt.show()

    def predict(self, samples):
        result = []
        for sample in samples:
            result.append(self.predict_sample(sample))
        return np.array(result)
    
    def accuracy(self, y_true, y_pred):
        i = 0
        hits = 0
        for pred in y_pred:
            if pred == y_true[i]:
                hits = hits + 1
        i = i + 1
        return (hits / len(y_true)) * 100    

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
def mapFeature(X1, X2, degrees = 6):
	out = np.ones( (np.shape(X1)[0], 1) )
	
	for i in range(1, degrees+1):
		for j in range(0, i+1):
			term1 = X1 ** (i-j)
			term2 = X2 ** (j)
			term  = (term1 * term2).reshape( np.shape(term1)[0], 1 ) 
			out   = np.hstack(( out, term ))
	return out  
