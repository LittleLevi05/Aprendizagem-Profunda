# Semana 1

Criação duma versão própria de uma *Deep Neuronal Network*.


![enter image description here](https://raw.githubusercontent.com/henriqueparola/Aprendizagem-Profunda/main/Semana_1/images/banner.png)
### Descrição
De maneira semelhante a biblioeteca de rede neuronal **Keras**, foi concebido em Python as classes *DNN* e *Layer*, que possuem as mesmas funcionalidades das classes *Sequential* e *Dense*, respectivamente.
* **Layer**: 
* **DNN**:

### Pipeline de execução

Exemplo de utilização das classes mencionadas. No caso de não querer explicitamente definir os pesos de cada conexão entre camadas:

````python
ds = Dataset("dataset.data")
model = DNN(ds)
model.add(Layer(nodes=2, activation='sigmoid'))
model.add(Layer(nodes=4, activation='sigmoid'))
model.add(Layer(nodes=3, activation='sigmoid'))
model.build_model()
model.predict(np.array([0,0]))
model.costFunction()
````

No caso de querer explicitamente definir os pesos de cada conexão entre camadas:

````python
ds = Dataset("xnor.data")
model = DNN(ds)
model.add(Layer(nodes=2, weights=np.array([[-30,20,20],[10,-20,-20]]), activation='sigmoid'))
model.add(Layer(nodes=2, weights=np.array([[-10,20,20]]), activation='sigmoid')) 
model.predict(np.array([1,1]))
model.costFunction()
````

### Testes
Para a execução de alguns testes sobre a DNN segundo diferentes formas de uso, basta executar o *python *tests.py**, ficheiro o qual contém testes dois tipos de testes:
* Teste do **XOR**, constituído de uma única camada oculta, e valores de **pesos** bem definidos passados explicitamente;
* Teste da função *build_model()*, constituído de uma gama variada de *layers* com um número variado de nodos em cada *layer* a ser experimentado. Neste teste **não** é passado valores de **pesos** explicitamente, uma vez que os mesmos serão descobertos através da função *build_model()*