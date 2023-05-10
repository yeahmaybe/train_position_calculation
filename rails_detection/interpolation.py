import cv2
import numpy as np
from srccam import CalibReader, Calib, Camera, Point3d
from pyar import Camera as pyarCamera, Size
from pyar import Point2D
from pyar import Point3D
import matplotlib.pyplot as plt


def getWheelAngle(spline) -> float:
    derivative = np.polyder(spline, 1)
    tan = derivative[0]
    wheel_angle = min(np.pi / 2 - np.arctan(tan), np.pi / 2 + np.arctan(tan))

    return wheel_angle


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


def get_3d_from_2d(pyar_camera, points) -> list[Point3D]:
    points_3d = []
    for point in points:
        point_3d = pyar_camera.reproject_point_with_height(point, 0)
        points_3d.append(point_3d)
    return points_3d


def get_spline(X, Y):
    weights = [1 / np.e ** (i ** 3 / 20) for i in range(len(Y))]
    print("Веса точек для регрессии:", weights)

    quad_polynom = np.polyfit(X, Y, 2, w=weights, full=True)
    line_polynom = np.polyfit(X, Y, 1, w=weights, full=True)

    # словарь {ошибка: полином}
    hypotheses = {
        quad_polynom[1][0]: quad_polynom[0],
        line_polynom[1][0]: line_polynom[0]
    }

    keys = list(hypotheses.keys())
    model = np.poly1d(hypotheses[min(keys)])
    return model


def draw_spline(image, pyar_camera, polyline, model):

    polyline_3d = np.array([(pt, model(pt), 0) for pt in polyline])
    for pt_3d in polyline_3d:
        pt_2d = pyar_camera.project_point(pt_3d)
        pt = (int(pt_2d.x), int(pt_2d.y))
        cv2.circle(image, pt, 2, (255, 0, 0), 2)

    cv2.imshow('result', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def interpolate(front_img):
    image = front_img
    file_name = '../data/city/leftImage.yml'
    wheel = (0, -4.6, 0)

    points = get_line_projection_points(image)
    pyar_camera = pyarCamera.from_yaml(file_name)

    points_3d = get_3d_from_2d(pyar_camera, points)
    points_3d = [(pt.x, pt.y, pt.z) for pt in points_3d]

    # отрисовка полученных точек осевой линиии
    for pt in points_3d:
        pr_pyar = pyar_camera.project_point(Point3D(pt))
        pr = tuple(map(int, [pr_pyar.x, pr_pyar.y]))
        cv2.circle(image, pr, 5, (255, 250, 20), 2)

    points_3d = [wheel] + points_3d
    points_on_surface = [(pt[0], pt[1]) for pt in points_3d]
    points_on_surface.sort()

    X = [pt[0] for pt in points_on_surface]
    Y = [pt[1] for pt in points_on_surface]

    model = get_spline(X, Y)
    print("Полином сплайна:", model, sep='\n')

    wheel_angle = getWheelAngle(model)
    print("Угол поворота передней тележки: ", wheel_angle / np.pi * 180, "градусов")

    plt.axis('equal')
    plt.scatter(X, Y)

    polyline = np.linspace(-1, 5, 50)
    plt.plot(polyline, model(polyline))
    plt.show()

    draw_spline(image, pyar_camera, polyline, model)

img = cv2.imread("../img.png")
img1 = cv2.imread("../img1.png")
img2 = cv2.imread("../img2.png")
img3 = cv2.imread("../img3.png")
img5 = cv2.imread("../img5.png")

interpolate(img5)
