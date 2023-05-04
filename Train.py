import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from Point import Point


class Train:
    def __init__(self, a, b, c, d, w, x0):
        self.x0 = x0
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.w = w

    def line(self, x, f, df):
        return df(self.x0) * (x - self.x0) + f(self.x0)

    def findP4(self, p3, c, f):
        N_iter = 100
        eps = 0.001
        dist = 0
        left = p3.x
        right = p3.x + c
        while abs(c * c - dist) > eps and N_iter > 0:
            mid = left / 2 + right / 2
            N_iter -= 1
            m = Point(mid, f(mid))
            dist = p3.dist2(m)
            if dist > c * c:
                right = mid
            else:
                left = mid

        return Point(mid, f(mid))

    def calculate_positions(self, x, f, df):

        global p1, p2, p3, p4, p5, v1, v2
        x1 = x + 1

        p2 = Point(x, f(x))
        v1 = Point(x1 - x, self.line(x1, f, df) - self.line(x, f, df)).norm()
        p1 = p2.subtract(v1.mult(self.a))
        p3 = p2.add(v1.mult(self.b))
        p4 = self.findP4(p3, self.c, f)
        v2 = p4.subtract(p3).norm()
        p5 = p4.add(v2.mult(self.d))

    def draw_back_rectangle(self):

        plt.gca().add_patch(Rectangle((p1.x, p1.y), self.w, self.a + self.b,
                                      angle=np.arctan(v1.y / v1.x) / np.pi * 180 - 90,
                                      edgecolor='blue',
                                      facecolor='none',
                                      lw=1))

        plt.gca().add_patch(Rectangle((p3.x, p3.y), self.w, self.a + self.b,
                                      angle=np.arctan(v1.y / v1.x) / np.pi * 180 + 90,
                                      edgecolor='blue',
                                      facecolor='none',
                                      lw=1))

    def draw_front_rectangle(self):

        plt.gca().add_patch(Rectangle((p3.x, p3.y), self.w, self.c + self.d,
                                      angle=np.arctan(v2.y / v2.x) / np.pi * 180 - 90,
                                      edgecolor='blue',
                                      facecolor='none',
                                      lw=1))

        plt.gca().add_patch(Rectangle((p5.x, p5.y), self.w, self.c + self.d,
                                      angle=np.arctan(v2.y / v2.x) / np.pi * 180 + 90,
                                      edgecolor='blue',
                                      facecolor='none',
                                      lw=1))

    def draw(self):
        self.draw_back_rectangle()
        self.draw_front_rectangle()
