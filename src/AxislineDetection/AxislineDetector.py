import cv2
import numpy as np
from skimage.morphology import medial_axis
from pyar import Point2D, Point3D
from src.Clasterization.KMeanClasterizer import clasterize2


def get_line_projection_points(image) -> list[Point2D]:
    def exclude_margin(coordinates):
        nearest_points = []
        for coor in coordinates:
            if coor[0][1] < image.shape[0] - 5:
                nearest_points.append((coor[0][0], coor[0][1]))
        return nearest_points

    thresh = get_rail_skeleton(image)
    pts = cv2.findNonZero(thresh)
    pts = exclude_margin(pts)
    pts.sort()
    pts = pts[::25]
    result = list(map(Point2D, pts))

    return result


def get_rail_skeleton(image):
    hsv_min = (70, 160, 192)
    hsv_max = (78, 255, 224)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    cv2.imshow("1", thresh)

    # thresh = cv2.blur(thresh, (50, 50))
    # _, thresh = cv2.threshold(thresh, 240, 255, 0)

    thresh = cv2.blur(thresh, (80, 80))
    cv2.imshow("2", thresh)

    _, thresh = cv2.threshold(thresh, 240, 255, 0)
    cv2.imshow("3", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    cv2.imshow("4", thresh)
    # thresh = cv2.dilate(thresh, kernel, iterations=5)
    # cv2.imshow("3.5", thresh)

    thresh = np.asarray(medial_axis(thresh), "uint8")

    return thresh


def get_surface_projections(pyar_camera, image) -> list[Point3D]:
    points = get_line_projection_points(image)
    wheel = Point3D(0, -4.6, 0)
    before_wheel = Point3D(0, -5, 0)

    get_3d = lambda p: pyar_camera.reproject_point_with_height(p, 0)
    points_3d = list(map(get_3d, points))
    points_3d = list(filter(lambda p: p.y < 5, points_3d))

    points_on_way = clasterize2(points_3d)
    points_on_way = [before_wheel, wheel] + points_on_way
    points_on_way.sort(key=lambda p: p.y)

    return points_on_way
