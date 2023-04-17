import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm(self):
        length = np.sqrt(self.x * self.x + self.y * self.y)
        return Point(self.x / length, self.y / length)

    def subtract(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def add(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def mult(self, a):
        return Point(self.x * a, self.y * a)

    def length(self):
        return np.sqrt(self.x * self.x + self.y * self.y)

    def length2(self):
        return self.x * self.x + self.y * self.y

    def dist2(self, p):
        return self.subtract(p).length2()

    def dist(self, p):
        return np.sqrt(self.dist2(p))

    def inversed(self):
        return self.mult(-1)

