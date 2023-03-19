# Semana 3

Criação duma versão própria de uma *Deep Neuronal Network*.


![enter image description here](https://raw.githubusercontent.com/henriqueparola/Aprendizagem-Profunda/main/Semana_3/images/banner.png)
### Descrição
De maneira semelhante a biblioeteca de rede neuronal **Keras**, foi concebido em Python as classes *DNN* e *Layer*, que possuem as mesmas funcionalidades das classes *Sequential* e *Dense*, respectivamente.
* **Layer**: classe que representa uma **camada** da rede. Deve-se criar não só as *hidden layers*, como também a camada de *input* da rede. Após a criação das camadas, estas devem ser utilizadas como parâmetros no método **add()** da classe DNN, de forma **ordenada**. Os parâmetros de criação de uma *Layer* são:
    * **nodos**: número de nodos da camada (número de *features* do *dataset* no caso da primeira camada).
    * **weights**: matriz Numpy com os valores dos pesos das conexões de cada nodo da camada a ser definida com os nodos da camada seguinte. Isto é, se estou definindo uma camada com 3 nodos, e de seguida haverá uma camada com 3 nodos, deve-se ter uma matrix **2x3**, em que a primeira linha da matriz contém o valor dos pesos da conexão de todos os nodos da camada com 2 nodos com o **primeiro** nodo da camada seguinte, e a segunda linha contém o valor dos pesos da conexão de todos os nodos da camada com 3 nodos com o **segundo** nodo da camada seguinte. Daí surge que, no caso de haver apenas uma camada de *ouput*, esta não precisa ser definida, uma vez que não há camadas para além delas, sendo inferido que a última camada criada terá conexão com a camada de *ouput* que tem apenas **um nodo**.
    * **activation**: função de ativação dos nodos da camada. Até então é são disponiblizadas as funções 'sigmoid' ou 'relu'.
* **DNN**: classe que representa a rede como um todo. No momento de sua criação é passado como parâmetro o *dataset* no qual o modelo será treinado. Os métodos disponibilizados são:
    * **add()**: adiciona uma *Layer* na rede. Deve-se ser executado de maneira ordenada. A primeira *Layer* é considerada a *input layer*.
    * **predict()**: prevê a *feature Y* de uma instância X.
    * **build_model()**: realiza o treinamento do modelo segundo o *dataset* passado como parâmetro no momento da criação da rede.
    * **cost_function()**: retorna o valor da *loss function* da rede. 

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
