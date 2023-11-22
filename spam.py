from nn import NN, g
import numpy as np
import matplotlib.pyplot as p

datafile = open("spambase.data", "r")
data = datafile.read().split("\n")[:-1]
data = np.array([[float(e) for e in i.split(",")] for i in data])

def normalize_column(c): return (c - c.min()) / (c.max() - c.min())
for i in range(54,57): data[:, i] = normalize_column(data[:, i])

data = [(np.array([i[:-1]]).T, np.array([[i[-1]]])) for i in data]

datafile.close()

n_attributes = len(data[0][0])
nn = NN([n_attributes, 1])

training_set = data[:int(len(data)*0.8)+1] 
test_set = data[int(len(data)*0.8)+1:]

err = nn.train(training_set, 0.00001, 100000, 0.01)

p.plot([e+1 for e in range(len(err))], err, label="error rate" + str(i+1))
p.legend()
p.show()

print(nn.error_rate(test_set))