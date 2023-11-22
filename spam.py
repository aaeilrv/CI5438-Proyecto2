from nn import NN, g
import numpy as np

datafile = open("spambase.data", "r")
data = datafile.read().split("\n")[:-1]
data = np.array([[float(e) for e in i.split(",")] for i in data])

def normalize_column(c): return (c - c.min()) / (c.max() - c.min())
for i in range(54,57): data[:, i] = normalize_column(data[:, i])

data = [(np.array([i[:-1]]).T, np.array([[i[-1]]])) for i in data]

datafile.close()

n_attributes = len(data[0][0])
nn = NN([n_attributes, 1], lambda x: g(x))
nn.train()