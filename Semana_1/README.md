# Semana 1

Criação duma versão própria do algoritmo de *Machine Learning* de **Regressão Logística**.


![enter image description here](https://raw.githubusercontent.com/henriqueparola/Aprendizagem-Profunda/main/Semana_1/images/banner.png)
### Descrição
De maneira semelhante ao módulo **LogisticRegression** do *sklearn.linear_model*, foi concebido em Python as funcionalidades de tal algortimo, assentando-se somente na biblioteca *NumPy*.  Deste modo foi preciso a criação de duas classes:
* **Dataset**: a partir de um ficheiro formatado por delimitadores (como um ficheiro csv), são armazenadas as variáveis independentes na instância X (*features* do dataset) e a variável dependente na instância **Y** (*target* a ser previsto, assumindo-se como sendo a última coluna do dataset), representadas na forma de *array* multidimensional do NumPy (ndarray). Para além disto, tal classe fornece duas importantes funcionalidades:
	*  *standardização* dos dados através do método *standardize(), armazenando em **Xst** as variáveis independentes *standardizadas*;
	* Separação de dados de treino e teste, a semelhança do método *train_test_split()* do módulo *model_selection* do *sklearn*. Tal método retorna uma tupla constituídas de quatro *ndarrays*: **x_train**,  **x_test**, **y_train**, **y_test**,  sendo **x** referente as variáveis independentes e **y** as *target features*. Tal como no *sklearn*, é possível inserir a percentagem dos dados de teste, assim como uma *seed*.
* **LogisticRegression**: implementa o algoritmo de regressão logística. Possui quatro alternativas de execução, utilizando:
	 * **Gradiente descendente** com ou sem **regularização** dos parâmetros;
	 *  **Métodos de optimizações númericas  avançados** com ou sem **regularização** recorrendo ao *package optimize*
Para o uso dos métodos de optimização invés do gradiente descendente, basta utilizar o parâmetro *optimize=**True*** no constructor do modelo. E no caso de se querer utilizar regularização, basta utilizar *regularization=**True***. No caso de se usar o Gradiente Descendente, ainda é possível construir o modelo com os parâmetros **alpha** e **iters** (número de iterações).  Os métodos mais relevantes desta classe são:
	 * *train(x_train, y_train)*: treina o modelo com os dados de treino passados. *x_train* contém as variáveis independentes num *ndarray* e *y_train* as variáveis *target*. Análogo ao método *fit()* do módulo *LogisticRegression* do *sklearn*.
	 * *predict(x_test)*: retorna um *ndarray* com as previsões da variável *target* das entradas passadas no *ndarray* x_test. Análogo ao método *predict()* do módulo *LogisticRegression* do *sklearn*.
	 * *accuracy(y_true, y_pred)*: retorna a *accuracy* obtida em relação aos dados previstos, *y_pred* e aos dados reais, *y_true*. Análogo ao método *accuracy_score()* do módulo *metrics* do *sklearn*.

### Pipeline de execução

Exemplo de utilização das classes mencionadas:
````python
    ds = Dataset("hearts-bin.data")
    x_train, x_test, y_train, y_test = ds.train_test_split(test_size=0.2, random_state=2023)
	logmodel = LogisticRegression(regularization= True, optimize=False, alpha=0.05, iters=10000)
    logmodel.train(x_train=x_train,y_train=y_train)
    y_pred = logmodel.predict(x_test)
    accuracy = logmodel.accuracy(y_test,y_pred)
````

