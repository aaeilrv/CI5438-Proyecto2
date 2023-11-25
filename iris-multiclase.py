from nn import NN
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
    mi = 0
    for i in range(len(value)):
        if value[i] > value[mi]: mi = i

    v = [0 for i in range(len(value))]
    for i in range(len(v)):
        if i == mi: v[i] = 1

    return v
    

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
    fp = 0
    fn = 0

    # Datos de entrenamiento
    train, test = split_dataset(dataframe)
    X_train = train.drop(columns=['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']).to_numpy()
    y_train = train[['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']].to_numpy()
    X_test = test.drop(columns=['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']).to_numpy()
    y_test = test[['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']].to_numpy()

    training_data = [(np.array([X_train[i]]).T.astype(float), np.array([y_train[i]]).T.astype(int)) for i in range(len(X_train))]
    testing_data = [(np.array([X_test[i]]).T.astype(float), np.array([y_test[i]]).T.astype(int)) for i in range(len(X_test))]


    # Crear la red neuronal
    red_neuronal = NN(tamano_red)

    # Entrenar la red neuronal
    red_neuronal.train(training_data, learning_rate, max_iteration, min_error)

    # Probar la red neuronal
    for x, y in testing_data:

        y = y.flatten()
        salida = red_neuronal.h(x)[1][-1].flatten()
        salida_normalizada = umbral_multiclase(salida)

        #print(f"Entrada: {x.flatten()}, Salida esperada: {y}, Salida de la red: {salida} -> {salida_normalizada}")
        
        if not check_error(salida_normalizada, y):
            error += 1

        fp += falsos_positivos(salida_normalizada, y)
        fn += falsos_negativos(salida_normalizada, y)

    print(f"Total tests: {len(X_test)}; Total tests fallados: {error}")
    print(f"Falsos positivos: {fp}")
    print(f"Falsos negativos: {fn}")

learning_rate = 0.01
max_iteration = 2000
min_error = 0.1
capa_oculta = [4, 5, 3]

print(capa_oculta)
for i in range(3):
    print(f'\nnum interaciones: {max_iteration}, learning rate: {learning_rate}, min error: {min_error}')
    clasificacion_multiclase(dataset, learning_rate, max_iteration, min_error, capa_oculta)