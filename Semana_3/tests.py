from dnn import DNN
from layer import Layer
import numpy as np
from dataset import Dataset

# Test different number of hidden nodes, layers and activation functions
def testBuildModel():
    ds = Dataset("xnor.data")
    print("> Initializing model ...")
    model = DNN(ds)
    print("> Adding first layer without weights with SIGMOID activation function ... ")
    model.add(Layer(nodes=2, activation='sigmoid'))
    print("> Adding second layer without weights with RELU activation function ... ")
    model.add(Layer(nodes=4, activation='relu'))
    print("> Adding third layer without weights with SIGMOID activation function ... ")
    model.add(Layer(nodes=3, activation='sigmoid'))
    print("> Building model ... ")
    model.build_model()
    print("> Making predictions ...")
    print("> Input: [0,0], Output: ", model.predict(np.array([0,0]) ) )
    print("> Input: [0,1], Output: ", model.predict(np.array([0,1]) ) )
    print("> Input: [1,0], Output: ", model.predict(np.array([1,0]) ) )
    print("> Input: [1,1], Output: ", model.predict(np.array([1,1]) ) )
    print("> Calculating Cost function")
    print("> Cost function result: ", model.costFunction())

# Test XOR statement with pre-defined weights 
def testXOR():
    ds = Dataset("xnor.data")
    print("> Initializing model ...")
    model = DNN(ds)
    print("> Adding first layer ... ")
    model.add(Layer(nodes=2, weights=np.array([[-30,20,20],[10,-20,-20]]), activation='sigmoid'))
    print("> Adding second layer ... ")
    model.add(Layer(nodes=2, weights=np.array([[-10,20,20]]), activation='sigmoid'))
    print("> Making predictions ...")
    print("> Input: [0,0], Output: ", model.predict(np.array([0,0]) ) )
    print("> Input: [0,1], Output: ", model.predict(np.array([0,1]) ) )
    print("> Input: [1,0], Output: ", model.predict(np.array([1,0]) ) )
    print("> Input: [1,1], Output: ", model.predict(np.array([1,1]) ) )
    print("> Calculating Cost function")
    print("> Cost function result: ", model.costFunction())

if __name__ == '__main__':
    testXOR()
    testBuildModel()