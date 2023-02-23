# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:02:12 2023

@author: parol
"""
from dataset import Dataset 
from logistic_regression import LogisticRegression

# Pipeline de execução com Gradiente Descendente sem métodos sofisticados e sem regulariação
def testGrad(filename):
    ds = Dataset(filename)
    print("> Realizando split do dataset ... ")
    dataset_train, dataset_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.05, iters=10000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)
  
# Pipeline de execução com Gradiente Descendente sem métodos sofisticados e com regulariação
def testGradReg(filename):
    ds = Dataset(filename)
    print("> Realizando split do dataset ... ")
    dataset_train, dataset_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization= True, optimize=False, alpha=0.05, iters=10000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)
    
# Pipeline de execução com Gradiente Descendente com métodos sofisticados e sem regulariação    
def testOpt(filename):
    ds = Dataset(filename)
    print("> Realizando split do dataset ... ")
    dataset_train, dataset_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization= False, optimize=True)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)
    
# Pipeline de execução com Gradiente Descendente com métodos sofisticados e com regulariação    
def testOptReg(filename):
    ds = Dataset(filename)
    print("> Realizando split do dataset ... ")
    dataset_train, dataset_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization= True, optimize= False)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)

# Testa as quatro diferentes formas de funcionamento do algoritmo da regressão logística:
# 1-) Gradiente Descendente sem métodos sofisticados e sem regulariação
# 2-) Gradiente Descendente sem métodos sofisticados e com regulariação
# 3-) Gradiente Descendente com métodos sofisticados e sem regulariação
# 4-) Gradiente Descendente com métodos sofisticados e com regulariação
def testAlgorithms(filename):
    print("> Iniciando Gradiente Descendente sem métodos sofisticados e sem regulariação ... ")
    testGrad(filename)
    print("> ---------------------------------------")
    print("> Iniciando Gradiente Descendente sem métodos sofisticados e com regulariação ... ")
    testGradReg(filename)
    print("> ---------------------------------------")
    print("> Iniciando Gradiente Descendente com métodos sofisticados e sem regulariação ... ")
    testOpt(filename)
    print("> ---------------------------------------")
    print("> Iniciando Gradiente Descendente com métodos sofisticados e com regulariação ... ")
    testOptReg(filename)
   
# Testa o desempenho do algorimto segundo quatro variações do tamanho do conjunto de testes:
# 1-) 80% treino, 20% teste
# 2-) 70% treino, 30% teste
# 3-) 60% treino, 40% teste
# 4-) 50% treino, 50% teste
def testSplit(filename): 
    ds = Dataset(filename)
    print("> Realizando split do dataset com 80% para treino ... ")
    dataset_train, dataset_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)
    
    ds = Dataset(filename)
    print("> Realizando split do dataset com 70% para treino ... ")
    dataset_train, dataset_test =  ds.train_test_split(test_size=0.3, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)
    
    ds = Dataset("log-ex1.data")
    print("> Realizando split do dataset com 60% para teste ... ")
    dataset_train, dataset_test = ds.train_test_split(test_size=0.4, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)
    
    ds = Dataset("log-ex1.data")
    print("> Realizando split do dataset com 50% para treino ... ")
    dataset_train, dataset_test = ds.train_test_split(test_size=0.5, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)
   
# Testa a standardização dos dados
def testStandardization(filename):
    print("> Teste sem standardização ... ")
    ds = Dataset(filename)
    print("> Realizando split do dataset com 80% para treino ... ")
    dataset_train, dataset_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)
    
    print("> Teste com standardização ...")
    
    ds = Dataset(filename)
    print("> Realizando split do dataset com 80% para treino ... ")
    dataset_train, dataset_test = ds.train_test_split(test_size=0.2, random_state=2023)
    
    dataset_train.standardize()
    
    print("> Inicializando modelo  ... ")
    logmodel = LogisticRegression(standardized=True, regularization=False, optimize=False, alpha=0.005, iters=200000)
    
    print("> Treinando o modelo com os dados de treino ... ")
    logmodel.train(dataset_train=dataset_train)
    
    print("> Realizando as previsões dos dados de teste ... ")
    y_pred = logmodel.predict(samples=dataset_test.X)
    
    print("> Calculando accuracy ... ")
    accuracy = logmodel.accuracy(y_true=dataset_test.Y,y_pred=y_pred)

    print("> accuracy: ", accuracy)
    
if __name__ == '__main__':
    #testAlgorithms("hearts-bin.data")
    #testSplit("hearts-bin.data")
    testStandardization("hearts-bin.data")

    