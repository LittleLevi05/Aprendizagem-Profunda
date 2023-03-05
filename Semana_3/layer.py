import numpy as np

class Layer:
    def __init__(self, nodes, weights=None, activation='sigmoid'):
        self.nodes = nodes 
        self.activation = activation
        self.W = weights

    # Layer execution function given an input (for predictions)
    def calculate(self, input):
        z3 = np.dot(self.W, input)
        a2 = np.empty([z3.shape[0] + 1])
        a2[0] = 1
        if self.activation == 'sigmoid':
            a2[1:] = sigmoid(z3)
        return a2
    
    # Layer execution function given an input (for last layer)
    def calculate2(self, input):
        z3 = np.dot(self.W, input)
        if self.activation == 'sigmoid':
            return sigmoid(z3)

    # Layer execution function given an input (for cost functions)
    def calculate3(self,input):
        Z2 = np.dot(input, self.W.T)
        ones = np.ones([Z2.shape[0],1])
        if self.activation == 'sigmoid':
            activationResult = sigmoid(Z2)

        A2 = np.hstack((ones, activationResult))
        return A2
    
    # Layer execution function given an input (for last layer in cost functions)
    def calculate4(self, input):
        Z2 = np.dot(input, self.W.T)
        if self.activation == 'sigmoid':
            activationResult = sigmoid(Z2)

        return activationResult

def sigmoid(x):
  return 1 / (1 + np.exp(-x))