import matplotlib.pyplot as plt
import numpy as np
from Train import Train


def f(x):
    return -0.1 * x * x + 4

def df(x):
    return -0.1 * 2 * x


a, b, c, d = [1, 2, 2, 1]
x0 = 0

for x0 in np.arange(0, 3, 0.1):
    train = Train(a, b, c, d, x0)
    train.calculate_positions(x0, f, df)

    X = np.arange(-10, 10, 0.01)
    Y = f(X)

    fig, ax = plt.subplots(1)

    plt.axis('equal')
    ax.set_xlim([-1, 6])
    ax.set_ylim([-1, 6])

    train.draw()

    plt.plot(X, Y)
    plt.show()
