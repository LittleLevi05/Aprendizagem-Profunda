# -*- coding: utf-8 -*-
"""
@author: miguelrocha
"""

import numpy as np
from dataset import Dataset
import matplotlib.pyplot as plt

class LogisticRegression:

    # Check
    def __init__(self, regularization = False, optimize = False, lamda = 1, alpha = 0.01, iters = 10000):
        self.theta = None
        self.regularization = regularization
        self.optimize = optimize
        self.lamda = lamda
        self.alpha = alpha
        self.iters = iters
        self.X = None
        self.Y = None
         
    # Check
    def printCoefs(self):
        print(self.theta)

    # Check
    def probability(self, instance):
        x = np.empty([self.X.shape[1]])
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        return sigmoid(np.dot(self.theta,x))

    # Check
    def predict_sample(self, instance):
        p = self.probability(instance)
        if p >= 0.5: res = 1
        else: res= 0 
        return res
  
    # Check
    def costFunction(self, theta = None):
        if theta is None: theta = self.theta
        m = self.X.shape[0]
        # predictions
        p = sigmoid(np.dot(self.X,theta))
        # cost function
        cost = (-self.y * np.log(p) - (1-self.y) * np.log(1-p))
        res = np.sum(cost) / m
        return res
        
    # Check
    def costFunctionReg(self, theta = None, lamda = 1):
        if theta is None: theta= self.theta
        m = self.X.shape[0]
        # predictions
        p = sigmoid(np.dot(self.X,theta))
        # cost function
        cost = (-self.y * np.log(p) - (1-self.y) * np.log(1-p))
        j = np.sum(cost) / m
        regTerm = (np.sum(np.square(self.theta))) * (self.lamda / (2 * self.X.shape[0]))
        res = j + regTerm
        return res

    # Check
    def gradientDescent(self, alpha = 0.01, iters = 10000):
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)  
        for its in range(iters):
            J = self.costFunction()
            # if its%1000 == 0: print(J)
            # delta 
            delta = self.X.T.dot(sigmoid(self.X.dot(self.theta)) - self.y)
            # self.theta 
            self.theta -= (alpha/m * delta)
            
    # Check
    def gradientDescentReg(self, alpha = 0.01, iters = 10000):
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)  
        for its in range(iters):
            J = self.costFunctionReg()
            # if its%1000 == 0: print(J)
            # delta 
            delta = self.X.T.dot(sigmoid(self.X.dot(self.theta)) - self.y)
            # self.theta 
            self.theta -= (alpha/m * delta)
            
    # Check
    def train(self, x_train, y_train):
        
        self.X = np.hstack((np.ones([x_train.shape[0],1]), x_train)) 
        self.y = y_train
        self.theta = np.zeros(self.X.shape[1])
        
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

    # Check
    def optim_model(self):
        from scipy import optimize

        n = self.X.shape[1]
        options = {'full_output': True, 'maxiter': 500}
        initial_theta = np.zeros(n)
        self.theta, _, _, _, _ = optimize.fmin(lambda theta: self.costFunction(theta), initial_theta, **options)
    
    # Check
    def optim_model_reg(self, lamda):
        from scipy import optimize

        n = self.X.shape[1]
        initial_theta = np.ones(n)        
        result = optimize.minimize(lambda theta: self.costFunctionReg(theta, lamda), initial_theta, method='BFGS', options={"maxiter":500, "disp":False} )
        self.theta = result.x    

    def mapX(self):
        self.origX = self.X.copy()
        mapX = mapFeature(self.X[:,1], self.X[:,2], 6)
        self.X = np.hstack((np.ones([self.X.shape[0],1]), mapX) )
        self.theta = np.zeros(self.X.shape[1])

    def plotModel(self):
        from numpy import r_
        pos = (self.y == 1).nonzero()[:1]
        neg = (self.y == 0).nonzero()[:1]
        plt.plot(self.X[pos, 1].T, self.X[pos, 2].T, 'k+', markeredgewidth=2, markersize=7)
        plt.plot(self.X[neg, 1].T, self.X[neg, 2].T, 'ko', markerfacecolor='r', markersize=7)
        if self.X.shape[1] <= 3:
            plot_x = r_[self.X[:,2].min(),  self.X[:,2].max()]
            plot_y = (-1./self.theta[2]) * (self.theta[1]*plot_x + self.theta[0])
            plt.plot(plot_x, plot_y)
            plt.legend(['class 1', 'class 0', 'Decision Boundary'])
        plt.show()

    def plotModel2(self):
        negatives = self.origX[self.y == 0]
        positives = self.origX[self.y == 1]
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

    # Check
    def predict(self, samples):
        result = []
        for sample in samples:
            result.append(self.predict_sample(sample))
        return np.array(result)
    
    # Check
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
  

# main - tests

def testGrad():
    ds = Dataset("hearts-bin.data")
    print("> Realizando split do dataset ... ")
    x_train, x_test, y_train, y_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.05, iters=10000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(x_train=x_train,y_train=y_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(x_test)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_test,y_pred)

    print("> accuracy: ", accuracy)
    
def testGradReg():
    ds = Dataset("hearts-bin.data")
    print("> Realizando split do dataset ... ")
    x_train, x_test, y_train, y_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization= True, optimize=False, alpha=0.05, iters=10000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(x_train=x_train,y_train=y_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(x_test)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_test,y_pred)

    print("> accuracy: ", accuracy)
    
def testOpt():
    ds = Dataset("hearts-bin.data")
    print("> Realizando split do dataset ... ")
    x_train, x_test, y_train, y_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization= False, optimize=True)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(x_train=x_train,y_train=y_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(x_test)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_test,y_pred)

    print("> accuracy: ", accuracy)
    
def testOptReg():
    ds = Dataset("hearts-bin.data")
    print("> Realizando split do dataset ... ")
    x_train, x_test, y_train, y_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization= True, optimize= False)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(x_train=x_train,y_train=y_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(x_test)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_test,y_pred)

    print("> accuracy: ", accuracy)

def testAlgorithms():
    print("> Iniciando Gradiente Descendente sem métodos sofisticados e sem regulariação ... ")
    testGrad()
    print("> ---------------------------------------")
    print("> Iniciando Gradiente Descendente sem métodos sofisticados e com regulariação ... ")
    testGradReg()
    print("> ---------------------------------------")
    print("> Iniciando Gradiente Descendente com métodos sofisticados e sem regulariação ... ")
    testOpt()
    print("> ---------------------------------------")
    print("> Iniciando Gradiente Descendente com métodos sofisticados e com regulariação ... ")
    testOptReg()
   
def testSplit(): 
    ds = Dataset("log-ex1.data")
    print("> Realizando split do dataset com 80% para treino ... ")
    x_train, x_test, y_train, y_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(x_train=x_train,y_train=y_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(x_test)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_test,y_pred)

    print("> accuracy: ", accuracy)
    
    ds = Dataset("log-ex1.data")
    print("> Realizando split do dataset com 70% para treino ... ")
    x_train, x_test, y_train, y_test = ds.train_test_split(test_size=0.3, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(x_train=x_train,y_train=y_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(x_test)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_test,y_pred)

    print("> accuracy: ", accuracy)
    
    ds = Dataset("log-ex1.data")
    print("> Realizando split do dataset com 60% para teste ... ")
    x_train, x_test, y_train, y_test = ds.train_test_split(test_size=0.4, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(x_train=x_train,y_train=y_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(x_test)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_test,y_pred)

    print("> accuracy: ", accuracy)
    
    ds = Dataset("log-ex1.data")
    print("> Realizando split do dataset com 50% para treino ... ")
    x_train, x_test, y_train, y_test = ds.train_test_split(test_size=0.5, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(x_train=x_train,y_train=y_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(x_test)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_test,y_pred)

    print("> accuracy: ", accuracy)
   
if __name__ == '__main__':
    testAlgorithms()
    # testSplit()

