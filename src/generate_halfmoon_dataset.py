import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from NeuralNet import NeuralNetwork


def generate_halfmoon_dataset(n_samples = 200, shuffle = True, noise = 0):
    np.random.seed(0)
    X, y = datasets.make_moons(n_samples, shuffle = shuffle, noise = noise)
    return X, y

X_train, y_train = generate_halfmoon_dataset(noise = 0.2)
X_test, y_test = generate_halfmoon_dataset(noise = 0.2)

train = [X_train, y_train]
nn = NeuralNetwork([2, 3, 3, 1], 0.05)
nn.train_network(train, n_epochs = 0, threshold = 0.01)

y_train_test = []
for i in range(len(y_train)):
    y_train_test.append(np.round(np.squeeze(nn.predict(X_train[i]))))

fig, axes = plt.subplots(1, 2, figsize = (7, 3))
axes[0].scatter(X_train[:,0], X_train[:,1], c=y_train, s=40)
axes[1].scatter(X_train[:,0], X_train[:,1], c=y_train_test, s=40)
fig.tight_layout()
plt.show()