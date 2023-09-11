import numpy as np

from pyar import Point2D, Point3D


def dist(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def clasterize2(points: list[Point3D]) -> list[Point3D]:

    point3d2tuple = lambda p: (p.x, p.y)
    tuple2ground = lambda tup: Point3D(tup[0], tup[1], 0)

    points = list(map(point3d2tuple, points))

    K = 2
    centers = []
    for k in range(K):
        centers.append(points[k])
    prev_centers = [(0, 0)] * K
    clusters = None
    while prev_centers != centers:
        clusters = []
        for _ in range(K):
            clusters.append([])
        prev_centers = centers.copy()
        for point in points:
            minDist = 999999
            idx = 0
            for i, center in enumerate(centers):
                if dist(point, center) < minDist:
                    minDist = dist(point, center)
                    idx = i
            clusters[idx].append(point)

        for c, center in enumerate(centers):
            mid_x = np.sum([pt[0] for pt in clusters[c]]) / len(clusters[c])
            mid_y = np.sum([pt[1] for pt in clusters[c]]) / len(clusters[c])
            centers[c] = (mid_x, mid_y)

    if dist(centers[0], centers[1]) < 1.5:
        return list(map(tuple2ground, clusters[0]+clusters[1]))

    minDist = 999999
    idx = 0
    for c, cluster in enumerate(clusters):
        for point in cluster:
            distance = dist((0, -4.6), point)
            if distance < minDist:
                minDist = distance
                idx = c
    return list(map(tuple2ground, clusters[idx]))
