#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
"""

import numpy as np
import matplotlib.pyplot as plt
from math import floor

class Dataset:
    
    # constructor
    def __init__(self, filename = None, X = None, Y = None):
        if filename is not None:
            self.readDataset(filename)
        elif X is not None and Y is not None:
            self.X = X
            self.Y = Y
        else:
            self.X = None
            self.Y = None
        
        self.Xst = None
        
    def readDataset(self, filename, sep = ","):
        data = np.genfromtxt(filename, delimiter=sep)
        self.X = data[:,0:-1]
        self.Y = data[:,-1]
        
    def getXy (self):
        return self.X, self.Y
    
    def nrows(self):
        return self.X.shape[0]
    
    def ncols(self):
        return self.X.shape[1]
    
    def standardize(self):
        self.mu = np.mean(self.X, axis = 0)
        self.Xst = self.X - self.mu
        self.sigma = np.std(self.X, axis = 0)
        self.Xst = self.Xst / self.sigma
    
    def plotData2vars(self, xlab, ylab, standardized = False):
        if standardized:
            plt.plot(self.Xst, self.Y, 'rx', markersize=7)
        else:
            plt.plot(self.X, self.Y, 'rx', markersize=7)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.show()
    
    def plotBinaryData(self):
        negatives = self.X[self.Y == 0]
        positives = self.X[self.Y == 1]
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim([self.X[:,0].min(), self.X[:,0].max()])
        plt.ylim([self.X[:,0].min(), self.X[:,0].max()])
        plt.scatter( negatives[:,0], negatives[:,1], c='r', marker='o', linewidths=1, s=40, label='y=0' )
        plt.scatter( positives[:,0], positives[:,1], c='k', marker='+', linewidths=2, s=40, label='y=1' )
        plt.legend()
        plt.show()
        
    def train_test_split(self, test_size, random_state):
        np.random.seed(random_state)
        test_indexes = np.random.choice(range(self.X.shape[0]),floor(len(self.X)*test_size),replace=False)
        train_indexes = [x for x in range(self.X.shape[0]) if x not in test_indexes ]
        x_train = self.X[train_indexes, :]
        x_test = self.X[test_indexes, :]
        y_train = self.Y[train_indexes]
        y_test = self.Y[test_indexes]
        dataset_train = Dataset(X=x_train,Y=y_train)
        dataset_test = Dataset(X=x_test,Y=y_test)
        return dataset_train, dataset_test
    
if __name__ == "__main__":
    def test():
        d = Dataset("lr-example1.data")
        d.plotData2vars("Population", "Profit")
        # print(d.getXy())
        # d.train_test_split(4,4)
    
    def testStandardized():
        d = Dataset("lr-example1.data")
        d.standardize()
        d.plotData2vars("Population", "Profit", True)
        print(d.getXy())
    
    def testBinary():
        ds= Dataset("log-ex1.data")   
        ds.plotBinaryData()   
        
    def testBinary2():
        ds= Dataset("log-ex2.data")   
        ds.plotBinaryData()   
    
    test()
    #testStandardized()   
    #testBinary()
    #testBinary2()
