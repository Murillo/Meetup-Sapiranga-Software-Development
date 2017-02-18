#Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

#Importando o dataset de exemplo
dataset = pd.read_csv("DataSet/racas_cachorros.csv")

#Agora, vamos separar as características e os rótulos que estão no DataSet e referenciá-los em duas variáveis.
#A variável X armazena as características do cachorro e a varável Y o seu rótulo.
X = dataset.iloc[:,:2]
y = dataset.iloc[:,2]

#Para testar e validar o modelo, vamos precisar separar os nossos dados.
#O método train_test_split() separa as informações em dados de treino e teste para as variáveis X e y
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5)

#Iniciar o modelo KNN da biblioteca sklearn
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_train, Y_train)

#Testando a acurácia do modelo
resultado = modelo.predict(X_test)
print(str(accuracy_score(resultado, Y_test) * 100) + "%")

#Classificando novos resultados que não estão no dataset
print("Shitzu" if modelo.predict([[5.2,21]]) == 0 else "Pastor Alemão")
print("Shitzu" if modelo.predict([[33.2,60]]) == 0 else "Pastor Alemão")
