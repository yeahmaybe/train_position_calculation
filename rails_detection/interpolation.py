import cv2
import numpy as np
from srccam import CalibReader, Calib, Camera, Point3d
from pyar import Camera as pyarCamera, Size
from pyar import Point2D
from pyar import Point3D
import matplotlib.pyplot as plt
from statistics import mean
from math import sin, cos, radians

from CircleEstimator import CircleEstimator
from PolynomeEstimator import PolynomeEstimator


def get_line_projection_points(image) -> list[Point2D]:
    hsv_min = (0, 200, 200)
    hsv_max = (50, 255, 255)
    kernel = np.ones((2, 2), 'uint8')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    cv2.imshow('thresh', thresh)

    non_zero_coordinates = cv2.findNonZero(thresh)
    pts = []
    for i, pt in enumerate(non_zero_coordinates):
        if i % 100 == 0:
            pts.append((pt[0][0], pt[0][1]))

    pts.sort()
    result = [Point2D(pt) for pt in pts]
    return result


def get_surface_projections(pyar_camera, points) -> list[Point2D]:
    wheel = (0, -4.6, 0)
    points_3d = []
    for point in points:
        point_3d = pyar_camera.reproject_point_with_height(point, 0)
        points_3d.append(point_3d)

    points_3d = [(pt.x, pt.y, pt.z) for pt in points_3d]

    points_3d = [wheel] + points_3d
    points_on_surface = [(pt[0], pt[1]) for pt in points_3d]
    points_on_surface.sort()
    points_on_surface = [Point2D(pt) for pt in points_on_surface]

    return points_on_surface


'''
        for pt in circle:
            pt = (round(pt[0], 2), round(pt[1], 2), 0)
            if (abs(pt[0]) == 0 and round(pt[1], 1) == -4.6) or pt[1] >= 0:
                prs_circ.append((pt[0], pt[1]))
                #pr_pyar = self.pyar_camera.project_point(Point3D(pt))
                #pr = tuple(map(int, [pr_pyar.x, pr_pyar.y]))
                #cv2.circle(image, pr, 4, (0, 0, 255), 3)

        #cv2.imshow('image', image)
'''


def draw_points(image, pyar_camera, points, color):
    polyline_3d = np.array([(pt[0], pt[1], 0) for pt in points])
    for pt_3d in polyline_3d:
        pt_2d = pyar_camera.project_point(pt_3d)
        pt = (int(pt_2d.x), int(pt_2d.y))
        cv2.circle(image, pt, 2, color, 2)


def chooseEstimator(surface_projections):
    circle_error = CircleEstimator().get_error(surface_projections)
    polynome_error = PolynomeEstimator().get_error(surface_projections)

    hypotheses = {
        circle_error: CircleEstimator(),
        #polynome_error: PolynomeEstimator()
    }
    least_error = min(hypotheses.keys())

    print("Ошибка на окружности: ", circle_error)
    print("Ошибка на полиномах: ", polynome_error)
    return hypotheses[least_error]


def interpolate(front_img):
    image = front_img
    file_name = '../data/city/leftImage.yml'

    points = get_line_projection_points(image)
    pyar_camera = pyarCamera.from_yaml(file_name)

    surface_projections = get_surface_projections(pyar_camera, points)

    prs = []
    for pt in surface_projections:
        pt = (pt.x, pt.y, 0)
        pr_pyar = pyar_camera.project_point(Point3D(pt))
        pr = tuple(map(int, [pr_pyar.x, pr_pyar.y]))
        prs.append(pr)
        cv2.circle(image, pr, 5, (255, 250, 20), 2)
        cv2.imshow('image', image)


    surface_projections = surface_projections[:10]


    estimator = chooseEstimator(surface_projections)
    wheel_angle = estimator.getWheelAngle(surface_projections) / np.pi * 180
    print("Угол поворота передней тележки: ", wheel_angle, "градусов")

    plt.axis('equal')

    X = [pt.x for pt in surface_projections]
    Y = [pt.y for pt in surface_projections]
    plt.scatter(X, Y)

    all_circle_points = CircleEstimator().get_points(surface_projections)
    all_poly_points = PolynomeEstimator().get_points(surface_projections)
    c_points = []
    p_points = []
    for pt in all_circle_points:
        if -30 <= pt[0] <= 30:
            c_points.append(pt)

    x = [pt[0] for pt in c_points]
    y = [pt[1] for pt in c_points]
    plt.plot(x, y)
    plt.show()

    draw_points(image, pyar_camera, all_circle_points, (200, 100, 100))
    #draw_points(image, pyar_camera, all_poly_points, (100, 200, 100))

    cv2.putText(image, "ANGLE: "+str(wheel_angle), (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow('result', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print("==============================")





img = cv2.imread("../img.png")
img3 = cv2.imread("../img3.png")
img5 = cv2.imread("../img5.png")

interpolate(img5)
interpolate(img3)
interpolate(img)




