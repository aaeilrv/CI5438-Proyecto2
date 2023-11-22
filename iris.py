import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv('iris.csv')

# Valores binarios para cada especie
iris_setosa = df.copy()
iris_setosa['species'] = iris_setosa['species'].map({'Iris-setosa': 1, 'Iris-versicolor': 0, 'Iris-virginica': 0})

iris_versicolor = df.copy()
iris_versicolor['species'] = iris_versicolor['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 0})

iris_virginica = df.copy()
iris_virginica['species'] = iris_virginica['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 1})

# Dividir los datasets en train y test balanceadamente:



iris_setosa_train, iris_setosa_test = train_test_split(iris_setosa, test_size=0.2)
iris_versicolor_train, iris_versicolor_test = train_test_split(iris_versicolor, test_size=0.2)
iris_virginica_train, iris_virginica_test = train_test_split(iris_virginica, test_size=0.2)

##### Clasificación Binaria #####

def umbral(value):
    return 1 if value >= 0.5 else 0

def clasificacion_binaria(dataframe, learning_rate, max_iteration, min_error, tamano_red):
    # Datos de entrenamiento
    X = dataframe.drop(columns=['species']).to_numpy()
    y = dataframe['species'].to_numpy()

    training_data = [(np.array([X[i]]).T.astype(float), np.array([y[i]]).astype(int)) for i in range(len(X))]

    # Crear la red neuronal
    red_neuronal = nn.NN([4, 1])

    # Entrenar la red neuronal
    red_neuronal.train(training_data, learning_rate, max_iteration, min_error)

    # Probar la red neuronal
    for x, y in training_data:
        print(f"Entrada: {x.flatten()}, Salida esperada: {y.flatten()}, Salida de la red: {red_neuronal.h(x)[1][-1].flatten()} -> {umbral(red_neuronal.h(x)[1][-1].flatten())}")

# Clasificación binaria con una única neurona
print("\nIris Setosa:")
clasificacion_binaria(iris_setosa_train, 0.1, 10000, 0.1, [4, 1])

print("\nIris versicolor:")

