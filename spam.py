from nn import NN
import numpy as np
import matplotlib.pyplot as p

import json
from pathlib import Path
import os
from functools import cmp_to_key

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from time import time


datafile = open("spambase.data", "r")
data = datafile.read().split("\n")[:-1]
data = np.array([[float(e) for e in i.split(",")] for i in data])

def normalize_column(c): return (c - c.min()) / (c.max() - c.min())
for i in range(54,57): data[:, i] = normalize_column(data[:, i])

data = [(np.array([i[:-1]]).T, np.array([[i[-1]]])) for i in data]

datafile.close()

n_attributes = len(data[0][0])
training_set = data[:int(len(data)*0.8)+1] 
test_set = data[int(len(data)*0.8)+1:]

def test_topologies():
    error_rates = []
    for depth in range(0,101, 10):
        for width in range(0, 101, 10):

            if depth == 0: depth = 1
            if width == 0: width = 1

            top = f"d{str(depth)}_w{str(width)}"
            print(top)

            nn = NN([n_attributes] + [width for i in range(depth)] + [1])

            err = nn.train(training_set, 0.00001, 1000, 0.1)

            dir = f"spam_tests/{top}"
            Path(f"{dir}").mkdir(parents=True, exist_ok=True) 
            file = open(f"{dir}/err.txt", "w")
            file.write(json.dumps(err))
            file.close()
            nn.save_weights(f"{dir}/model")

            error_rates.append((top, err[-1]))

    for e in error_rates: print(f"{e[0]}\t{str(e[1])} in {str(len(e[1]))} it")

def graph_topologies():

    tops = []

    for top_dir in os.listdir("spam_tests"):
        err_file = open(f"spam_tests/{top_dir}/err.txt", "r")
        tops.append({"top": top_dir, "err":json.loads(err_file.read())})
        err_file.close()

    def cmp(a,b): return a["err"][-1] - b["err"][-1] 
    sorted_tops = sorted(tops, key=cmp_to_key(cmp))

    for t in sorted_tops: print(f"error {t['top']}\t{str(t['err'][-1])} en {str(len(t['err']))} it")
    toptop = sorted_tops[0] 
    print("mejor topologia", toptop["top"]) #d20_w80


    fig, ax = p.subplots(subplot_kw={"projection": "3d"})
    X = np.concatenate((np.array([1]), np.arange(10, 110, 10)))
    Y = np.concatenate((np.array([1]), np.arange(10, 70, 10)))
    X, Y = np.meshgrid(X, Y)
    Z = np.reshape([t["err"][-1] for t in tops[:-3]], (-1, 11))
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    p.show()

def train_model():

    nn = NN(f"spam_train/model5")

    t0 = time()
    err = nn.train(training_set, 0.00006, 500, 0.1)
    t1 = time()
    print(t1-t0)

    nn.save_weights("spam_train/model6")
    file = open(f"spam_train/err6.txt", "w")
    file.write(json.dumps(err))
    file.close()

    #sesiones de entrenamiento
    #iteraciones    tasa de aprendizaje
    #50             0.00001
    #500            0.00001
    #500            0.000005
    #500            0.00002
    #500            0.00003
    #500            0.00006


def graph_training():
    rates = [0.00001, 0.000005, 0.00002, 0.00003, 0.00006]

    it = 0
    for i in range(len(rates)):
        err_file = open(f"spam_train/err{str(i+2)}.txt", "r")
        err =json.loads(err_file.read())
        err_file.close()

        p.plot([e for e in range(it, it+len(err))], err, label=f"tasa {int(rates[i]*1000000)}e-6")

        it += len(err)

    p.legend()
    p.show()


#test_topologies()
#train_model()
#graph_topologies()
graph_training()