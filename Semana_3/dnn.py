import numpy as np
from layer import Layer
from scipy import optimize

class DNN:
    def __init__(self, dataset):
        self.layers = []
        self.X, self.y = dataset.getXy()
        self.X = np.hstack ( (np.ones([self.X.shape[0],1]), self.X ) )

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, instance):
        x = np.empty([len(instance)+1])        
        x[0] = 1
        x[1:] = np.array(instance[:len(instance)])
        
        # First layer: dot product of their W with the instance input
        nextLayerInput = self.layers[0].calculate(x)
        
        # Each layer made de dot product with their W and the sigmoid function applied to the
        # result of the previous iteration calculation 
        for layer in self.layers[1:-1]:
            nextLayerInput = layer.calculate(nextLayerInput) 

        # Last layer
        return self.layers[-1].calculate2(nextLayerInput)

    def costFunction(self, weights=None):
        if weights is not None:
            floorLimit = 0
            upperLimit = 0
            iter = 0
            for layer in self.layers[:-1]:
                upperLimit += (layer.nodes+1) * self.layers[iter+1].nodes
                layer.W = weights[floorLimit:upperLimit].reshape([self.layers[iter+1].nodes, layer.nodes+1])
                floorLimit = upperLimit
                iter = iter + 1
            self.layers[-1].W = weights[floorLimit:].reshape([1, self.layers[-1].nodes+1])
                
        # First layer
        nextLayerInput = self.layers[0].calculate3(self.X)
        
        # Layer 1 ... Layer (len(self.layers) - 1)
        for layer in self.layers[1:-1]:
            nextLayerInput = layer.calculate3(nextLayerInput)

        # Last layer
        predictions = self.layers[-1].calculate4(nextLayerInput)

        m = self.X.shape[0]
        sqe = (predictions - self.y.reshape(m,1)) ** 2
        res = np.sum(sqe) / (2*m)
        return res
    
    def build_model(self):
        size = 0
        iter = 0
        # get the total number of weights
        for layer in self.layers[:-1]:
            size += (layer.nodes+1) * self.layers[iter+1].nodes
            iter = iter + 1

        size += self.layers[-1].nodes + 1

        initial_w = np.random.rand(size)        
        result = optimize.minimize(lambda w: self.costFunction(w), initial_w, method='BFGS', 
                                    options={"maxiter":1000, "disp":False} )
        
        weights = result.x
        floorLimit = 0
        upperLimit = 0
        iter = 0
        for layer in self.layers[:-1]:
            upperLimit += (layer.nodes+1) * self.layers[iter+1].nodes
            layer.W = weights[floorLimit:upperLimit].reshape([self.layers[iter+1].nodes, layer.nodes+1])
            floorLimit = upperLimit
            iter = iter + 1

        self.layers[-1].W = weights[floorLimit:].reshape([1, self.layers[-1].nodes+1])