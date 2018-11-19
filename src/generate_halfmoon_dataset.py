import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

def generate_halfmoon_dataset(n_samples = 200, shuffle = True, noise = 0):
    np.random.seed(0)
    X, y = datasets.make_moons(n_samples, shuffle = shuffle, noise = noise)
    plt.scatter(X[:,0], X[:,1], c=y, s=40)
    plt.show()

generate_halfmoon_dataset(noise = 0.2)