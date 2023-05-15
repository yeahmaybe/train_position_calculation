import cv2
import numpy as np
from srccam import CalibReader, Calib, Camera, Point3d
from pyar import Camera as pyarCamera, Size
from pyar import Point2D
from pyar import Point3D
import matplotlib.pyplot as plt
from statistics import mean
from math import sin, cos, radians


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

def get_circle(surface_projections):
    data = []

    for pt in surface_projections:
        data.append((pt.x, pt.y))
    print(data)
    minimum = {}
    points = [[data[i], data[j], data[k]] for i in range(len(data) - 2) for j in range(i + 1, len(data) - 1) for k in range(j + 1, len(data))]
    for z in points:
        x1, y1 = z[0]
        x2, y2 = z[1]
        x3, y3 = z[2]
        try:
            y0 = (2 * (x1 - x3) * (y2 ** 2 - y1 ** 2 - x1 ** 2 + x2 ** 2) + 2 * (x1 - x2) * (
                    x1 ** 2 - x3 ** 2 - y3 ** 2 + y1 ** 2)) / (
                         4 * (y2 - y1) * (x1 - x3) - 4 * (x1 - x2) * (y3 - y1))
        except ZeroDivisionError:
            continue
        if x1 - x3 != 0:
            x0 = ((x1 ** 2 - x3 ** 2 - y3 ** 2 + y1 ** 2) + 2 * y0 * (y3 - y1)) / (2 * (x1 - x3))
        if x1 - x2 != 0:
            x0 = ((x1 ** 2 - x2 ** 2 - y2 ** 2 + y1 ** 2) + 2 * y0 * (y2 - y1)) / (2 * (x1 - x2))
        R = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** (1 / 2)

        n1, n2, n3 = 0, 0, 0

        for t in data:
            if round((t[0] - x0) ** 2 + (t[1] - y0) ** 2, 3) < round(R ** 2, 3):
                n1 += 1
            elif round((t[0] - x0) ** 2 + (t[1] - y0) ** 2, 3) > round(R ** 2, 3):
                n2 += 1
            else:
                n3 += 1
        minimum[abs(n1 - n2)] = [
            '(x - {:.1f}) ** 2 + (y - {:.1f}) ** 2 = {:.1f} ** 2'.format(x0, y0, R).replace('- -', '+ '), z[0], z[1],
            z[2], n1, n2, n3, R, x0, y0]

    print('Окружность:', minimum[min(minimum)][0])
    print('Три точки, определяющие окружность:', str(minimum[min(minimum)][1]) + ',',
          str(minimum[min(minimum)][2]) + ',', minimum[min(minimum)][3])
    print('Радиус окружности:', minimum[min(minimum)][7])
    print('Начальные координаты:', str(minimum[min(minimum)][8]) + ', ' + str(minimum[min(minimum)][9]))

    a0 = minimum[min(minimum)][8]
    b0 = minimum[min(minimum)][9]
    R = minimum[min(minimum)][7]

    points = []

    for i in range(360):
        new_i = radians(i)
        si = sin(new_i)
        x = a0 + R * cos(new_i)
        y = b0 + R * sin(new_i)
        points.append((x, y))

    return points

def acc_calculate(prs, circle):
    acc = []
    for pt in prs:
        delt_x = circle[8] - pt[0]
        delt_y = circle[9] - pt[1]
        dif = ((delt_x ** 2 + delt_y ** 2) ** (1 / 2))
        dif = dif - circle[7]
        dif = abs(dif / circle[7])
        acc.append(dif)
    acc_mean = 1 - mean(acc)
    print("Точность аппроксимации - ", acc_mean)

def interpolate(front_img):
    image = front_img
    file_name = '../data/city/leftImage.yml'

    points = get_line_projection_points(image)
    pyar_camera = pyarCamera.from_yaml(file_name)

    surface_projections = get_surface_projections(pyar_camera, points)

    for pt in surface_projections:
        pt = (pt.x, pt.y, 0)
        pr_pyar = pyar_camera.project_point(Point3D(pt))
        pr = tuple(map(int, [pr_pyar.x, pr_pyar.y]))
        cv2.circle(image, pr, 5, (255, 250, 20), 2)
        cv2.imshow('image', image)

    X = [pt.x for pt in surface_projections]
    Y = [pt.y for pt in surface_projections]

    circle = get_circle(surface_projections)

    prs = []

    for pt in circle:
        pt = (pt[0], pt[1], 0)
        if (pt[0] > 0 and pt[1] > 0):
            pr_pyar = pyar_camera.project_point(Point3D(pt))
            pr = tuple(map(int, [pr_pyar.x, pr_pyar.y]))
            cv2.circle(image, pr, 4, (0, 0, 255), 3)
            prs.append(pr)

    cv2.imshow('image', image)


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

interpolate(img2)
