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
if (not os.path.isfile("nn_halfmoon_noise_0.1_tanh.npy")):
    train = [X_train, y_train]
    nn.train_network(train, n_epochs = 0, threshold = 0.001)
    np.save("nn_halfmoon_noise_0.1_tanh", nn.get_network())
else :
    W = np.load("nn_halfmoon_noise_0.1_tanh.npy")
    print("loaded weight matrix W = %s\n" %(W))
    nn.load_network(W)

y_test_test = []
for i in range(len(y_test)):
    y_test_test.append(np.around(np.squeeze(nn.predict(X_test[i]))))

y_train_test = []
for j in range(len(y_test)):
    y_train_test.append(np.around(np.squeeze(nn.predict(X_train[j]))))


plt.subplot(221)
plt.title("Train Data")
plt.scatter(X_test[:,0], X_test[:,1], c=y_train, s=40)

plt.subplot(222)
plt.title("Testing Training Data")
plt.scatter(X_train[:,0], X_train[:,1], c=y_train_test, s=40)

plt.subplot(223)
plt.title("Test Data")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, s=40)

plt.subplot(224)
plt.title("Testing test Data")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test_test, s=40)


plt.subplots_adjust(hspace = 0.4, wspace = 0.4)
plt.show()