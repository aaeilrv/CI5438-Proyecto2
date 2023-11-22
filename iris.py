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
def split_dataset(dataset):
    train, test = train_test_split(dataset, test_size=0.2)
    while train['species'].value_counts().min() < 10 or test['species'].value_counts().min() < 10:
        train, test = train_test_split(dataset, test_size=0.2)
    return train, test

iris_setosa_train, iris_setosa_test = train_test_split(iris_setosa, test_size=0.2)
#iris_versicolor_train, iris_versicolor_test = train_test_split(iris_versicolor, test_size=0.2)
#iris_virginica_train, iris_virginica_test = train_test_split(iris_virginica, test_size=0.2)

##### Clasificación Binaria #####

def umbral(value):
    return 1 if value >= 0.5 else 0

def clasificacion_binaria(dataframe, learning_rate, max_iteration, min_error, tamano_red):
    error = 0
    falsos_positivos = 0
    falsos_negativos = 0

    # Datos de entrenamiento
    train, test = split_dataset(dataframe)
    X_train = train.drop(columns=['species']).to_numpy()
    y_train = train['species'].to_numpy()
    X_test = test.drop(columns=['species']).to_numpy()
    y_test = test['species'].to_numpy()

    training_data = [(np.array([X_train[i]]).T.astype(float), np.array([y_train[i]]).astype(int)) for i in range(len(X_train))]
    testing_data = [(np.array([X_test[i]]).T.astype(float), np.array([y_test[i]]).astype(int)) for i in range(len(X_test))]

    # Crear la red neuronal
    red_neuronal = nn.NN(tamano_red)

    # Entrenar la red neuronal
    red_neuronal.train(training_data, learning_rate, max_iteration, min_error)

    # Probar la red neuronal
    for x, y in testing_data:
        print(f"Entrada: {x.flatten()}, Salida esperada: {y.flatten()}, Salida de la red: {red_neuronal.h(x)[1][-1].flatten()} -> {umbral(red_neuronal.h(x)[1][-1].flatten())}")
        if y.flatten() != umbral(red_neuronal.h(x)[1][-1].flatten()):
            error += 1
        if y.flatten() == 1 and umbral(red_neuronal.h(x)[1][-1].flatten()) == 0:
            falsos_negativos += 1
        if y.flatten() == 0 and umbral(red_neuronal.h(x)[1][-1].flatten()) == 1:
            falsos_positivos += 1

    print(f"Total tests: {len(X_test)}; Total tests fallados: {error}")
    print(f"Falsos positivos: {falsos_positivos}")
    print(f"Falsos negativos: {falsos_negativos}")

# Clasificación binaria con una única neurona

#### NOTA: en vez de hacerlo en tres separados, puedo hacerlo en uno solo donde las salidas esperadas sean del tipo:
# [1, 0, 0]; [0, 1, 0] y [0, 0, 1]

print("\nIris Setosa:")
clasificacion_binaria(iris_setosa, 0.1, 10000, 0.1, [4, 1])

#print("\nIris versicolor:")

