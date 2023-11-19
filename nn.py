import numpy as np
from math import exp
import json

from random import randint as r

#funcion de activacion
g = np.vectorize(lambda x: 1/(1+exp(-x)))

#derivada de la funcion de activacion para el descenso de gradiente
def gprime(X):
    return g(X)*(1-g(X))

class NN(object):

    #el parametro de entrada es una lista de enteros, donde cada enterode índice i representa
    #el tamaño de la capa i de la red neuronal. Debe tener al menos dos elementos representando
    #las capas de entrada y salida
    #tambien se puede inicializar con el nombre de un archivo de texto que guarda una lista de pesos
    def __init__(self, len_layers):
        if type(len_layers) == str:
            self.load_weights(len_layers)
        elif len(len_layers) < 2: raise IndexError("Debe definirse tamaño de capas de entrada y salida")
        else:
            self.weights = [np.random.randn(len_layers[i], len_layers[i-1]) for i in range(1,len(len_layers))]
            self.biases = [np.random.randn(L, 1) for L in len_layers[1:]]


    #funcion hipótesis de la red neuronal, toma como elemento
    #un valor de entrada y retorna un valor estimado de salidam,
    #asi como los valores intermedios de las capas escondidas, tanto los
    #valores originales como los valores resultantes de aplicar la funcion de activacion
    def h(self, X):
        Yh = X
        hidden_Y = [X]
        hidden_activated_Y = [X]
        for i in range(len(self.weights)):
            Yh = np.dot(self.weights[i], Yh) + self.biases[i]
            hidden_Y.append(Yh)
            if i != len(self.weights) -1: Yh = g(Yh)
            hidden_activated_Y.append(Yh)
        return hidden_Y, hidden_activated_Y
    

    def error_rate(self, examples):
        E = 0
        for ex in examples:
            E += (self.h(ex[0])[1][-1] - ex[1])**2
        return E[0]/len(examples)

    
    #funcion que calcula el gradiente de los pesos de la red en base a un
    #valor de entrada X y un valor de salida Y, utilizando la funcion de error
    #cuadrático como función de pérdida
    def gradient_descent(self, X, Y):

        hidden_Y, hidden_activated_Y = self.h(X)
        
        delta = Y - hidden_activated_Y[-1]
        
        weight_gradient = []
        bias_gradient = []

        #agregamos una matriz de pesos dummy para poder incluir el primer paso en la iteracion
        self.weights.append(np.identity(delta.size))

        for i in range(1,len(self.weights)):
            delta = np.dot(self.weights[-i].T, delta)*gprime(hidden_Y[-i])
            layer_gradient = delta*hidden_activated_Y[-i-1].T

            bias_gradient.insert(0,delta)
            weight_gradient.insert(0,layer_gradient)

        self.weights.pop()

        return weight_gradient, bias_gradient
    
    #función que realiza el algoritmo de backpropagation de error de un lote de ejemplos en base a 
    #una tasa de aprendizaje
    def backpropagation(self, examples, learning_rate):
        dldw = [np.zeros_like(L) for L in self.weights] 
        dldb = [np.zeros_like(L) for L in self.biases] 
        for (X,Y) in examples: 
            weight_gradient, bias_gradient = self.gradient_descent(X, Y)
            dldw = [i + e for i, e in zip(dldw, weight_gradient)]
            dldb = [i + e for i, e in zip(dldb, bias_gradient)]
            
        self.weights = [W + learning_rate*gradient/len(examples) for W, gradient in zip(self.weights, dldw)]
        self.biases = [B + learning_rate*gradient/len(examples) for B, gradient in zip(self.biases, dldb)]

    #funcion para entrenar al modelo en base a una serie de modelos, tasa de aprendizaje, 
    #y limites de iteracion y error
    def train(self, examples, learning_rate, max_iteration, min_error):
        for i in range(max_iteration):
            if self.error_rate(examples) <= min_error: break
            self.backpropagation(examples, learning_rate)
            print(self.error_rate(examples))

    #funcion que guarda el modelo actual en un archivo de texto
    def save_weights(self, filename):
        file = open(f"{filename}.txt", "w")
        data = {"weights":self.weights, "biases": self.biases}
        file.write(json.dumps(data))
        file.close()

    #funcion quer carga el modelo en un archivo de texto
    def load_weights(self, filename):
        file = open(f"{filename}.txt", "r")
        data = json.loads(file.read())
        self.weights = data.weights
        self.biases = data.biases
        file.close()


nn = NN([1, 10, 10, 10, 10, 10, 10, 10, 10, 1])

def f(X):
    return X*X

examples = [(X, f(X)) for X in [ np.vectorize(lambda x: int(x))(np.random.rand(1,1)*20) for i in range(1000)] ]
learning_rate = 0.01
max_iteration = 100000
min_error = 0.1

nn.train(examples,learning_rate, max_iteration, min_error)
        