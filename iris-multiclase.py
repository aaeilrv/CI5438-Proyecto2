import nnmulticlase as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv('iris.csv')

# Dividir los datasets en train y test balanceadamente:
def split_dataset(dataset):
    train, test = train_test_split(dataset, test_size=0.2)
    return train, test

dataset = pd.get_dummies(df, columns=['species'])

def umbral_multiclase(value):
    for i in range(len(value)):
        if value[i] >= 0.5:
            value[i] = 1
        else:
            value[i] = 0
    return value.astype(int)

def check_error(value, expected):
    for i in range(len(value)):
        if value[i] != expected[i]:
            return False
    return True

def falsos_positivos(value, expected):
    count = 0
    for i in range(len(value)):
        if value[i] == 1 and expected[i] == 0:
            count += 1
    return count

def falsos_negativos(value, expected):
    count = 0
    for i in range(len(value)):
        if value[i] == 0 and expected[i] == 1:
            count += 1
    return count

def clasificacion_multiclase(dataframe, learning_rate, max_iteration, min_error, tamano_red):
    error = 0
    falsos_positivos = 0
    falsos_negativos = 0

    # Datos de entrenamiento
    train, test = split_dataset(dataframe)
    X_train = train.drop(columns=['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']).to_numpy()
    y_train = train[['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']].to_numpy()
    X_test = test.drop(columns=['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']).to_numpy()
    y_test = test[['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']].to_numpy()

    training_data = [(np.array([X_train[i]]).T.astype(float), np.array([y_train[i]]).astype(int)) for i in range(len(X_train))]
    testing_data = [(np.array([X_test[i]]).T.astype(float), np.array([y_test[i]]).astype(int)) for i in range(len(X_test))]

    # Crear la red neuronal
    red_neuronal = nn.NN(tamano_red)

    # Entrenar la red neuronal
    red_neuronal.train(training_data, learning_rate, max_iteration, min_error)

    # Probar la red neuronal
    for x, y in testing_data:
        print(f"Entrada: {x.flatten()}, Salida esperada: {y.flatten()}, Salida de la red: {red_neuronal.h(x)[1][-1].flatten()} -> {umbral_multiclase(red_neuronal.h(x)[1][-1].flatten())}")
        
        if not check_error(umbral_multiclase(red_neuronal.h(x)[1][-1].flatten()), y.flatten()):
            error += 1

        falsos_positivos += falsos_positivos(umbral_multiclase(red_neuronal.h(x)[1][-1].flatten()), y.flatten())
        falsos_negativos += falsos_negativos(umbral_multiclase(red_neuronal.h(x)[1][-1].flatten()), y.flatten())

    print(f"Total tests: {len(X_test)}; Total tests fallados: {error}")
    print(f"Falsos positivos: {falsos_positivos}")
    print(f"Falsos negativos: {falsos_negativos}")

learning_rate = 0.1
max_iteration = 2000
min_error = 0.1
capa_oculta = [4, 1, 3]

for i in range(3):
    print(f'\nnum interaciones: {max_iteration}, learning rate: {learning_rate}, min error: {min_error}')

    print("\nMulticlase:")
    clasificacion_multiclase(dataset, learning_rate, max_iteration, min_error, capa_oculta)