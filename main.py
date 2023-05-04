import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
from Train import Train
import cv2


def f(x):
    return 1 - x*x/2 + x**4/24 - x**6/720 + x**8/40320


def df(x):
    return -x + x**3/6 - x**5/120 + x**7/5040


a, b, c, d = [0.3, 0.7, 0.7, 0.3]
w = 0.1


def create_frame(t):
    X0 = np.arange(-5, 5, 0.05)
    x0 = X0[t]

    train = Train(a, b, c, d, w, x0)
    train.calculate_positions(x0, f, df)

    X = np.arange(-10, 10, 0.01)
    Y = f(X)

    fig, ax = plt.subplots(1)
    plt.axis('equal')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-1, 3])

    train.draw()

    plt.plot(X, Y)
    plt.gcf().canvas.flush_events()
    plt.savefig(f'img_{t}.png',
                transparent=False,
                facecolor='white'
                )


time = list(map(int, np.arange(0, 10, 0.05)*20))
for t in time:
    create_frame(t)

frames = []
for t in time:
    img = cv2.imread(f'img/img_{t}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img).reshape(480, 640)
    frames.append(img)

imageio.mimsave('./train.gif',  # output gif
                frames,  # array of input frames
                fps=40)
