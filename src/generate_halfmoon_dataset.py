import numpy as np
import os.path
from sklearn import datasets
from matplotlib import pyplot as plt
from NeuralNet import NeuralNetwork


def generate_halfmoon_dataset(n_samples = 200, shuffle = True, noise = 0):
    np.random.seed(0)
    X, y = datasets.make_moons(n_samples, shuffle = shuffle, noise = noise)
    return X, y

X_train, y_train = generate_halfmoon_dataset(noise = 0.1)
X_test, y_test = generate_halfmoon_dataset(noise = 0.1)

nn = NeuralNetwork([2, 4, 2, 1], 0.03)
if (not os.path.isfile("nn_halfmoon_noise_0.1.npy")):
    train = [X_train, y_train]
    nn.train_network(train, n_epochs = 0, threshold = 0.01)
    np.save("nn_halfmoon_noise_0.1", nn.get_network())
else :
    W = np.load("nn_halfmoon_noise_0.1.npy")
    print(W)
    nn.load_network(W)

y_train_test = []
for i in range(len(y_train)):
    y_train_test.append(np.squeeze(nn.predict(X_train[i])))

y_test_test = []
for j in range(len(y_test)):
    y_test_test.append((np.squeeze(nn.predict(X_test[i])))) #its not the nn that's training incorrectly, something's not getting reinitialized after predict


plt.subplot(221)
plt.title("Training Data")
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, s=40)

plt.subplot(222)
plt.title("Testing Training Data")
plt.scatter(X_train[:,0], X_train[:,1], c=y_train_test, s=40)

plt.subplot(223)
plt.title("Test Data")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, s=40)

plt.subplot(224)
plt.title("Testing Test Data")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test_test, s=40)

plt.subplots_adjust(hspace = 0.4, wspace = 0.4)
plt.show()