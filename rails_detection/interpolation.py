import cv2
import numpy as np
from srccam import CalibReader, Calib, Camera, Point3d
from pyar import Camera as pyarCamera, Size
from pyar import Point2D
from pyar import Point3D
import matplotlib.pyplot as plt


def getWheelAngle(front_img):
    def getLineProjectionPoints(image):
        hsv_min = (0, 200, 200)
        hsv_max = (50, 255, 255)
        kernel = np.ones((2, 2), 'uint8')

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, hsv_min, hsv_max)
        #thresh = cv2.erode(thresh, kernel, iterations=1)
        cv2.imshow('thresh', thresh)

        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 0, minLineLength=10, maxLineGap=20)
        pts = [(line[0][0], line[0][1]) for line in lines]
        pts.sort()
        print(pts)
        #for pt in pts:
            #cv2.circle(image, (pt[0], pt[1]), 2, (255, 0, 0), 2)
        return pts


    image = front_img
    points = getLineProjectionPoints(image)
    points_3d = []

    file_name = '../data/city/leftImage.yml'
    pyar_camera = pyarCamera.from_yaml(file_name)
    #print(points)
    for point in points:
        point_2d = Point2D(point)
        point_3d = pyar_camera.reproject_point_with_height(point_2d, 0)
        tuple_point_3d = (point_3d.x, point_3d.y, point_3d.z)
        points_3d.append(tuple_point_3d)
    print(*points_3d, sep='\n')

    pts = points_3d

    for pt in pts:
        pr_pyar = pyar_camera.project_point(Point3D(pt))
        pr = tuple(map(int, [pr_pyar.x, pr_pyar.y]))
        cv2.circle(image, pr, 5, (255, 250, 20), 2)

    points_3d = pts
    #print(*points_3d, sep='\n')

    wheel = (0, -4.6, 0)
    points_3d = [wheel] + points_3d
    points_on_surface = [(pt[0], pt[1]) for pt in points_3d]
    points_on_surface.sort()

    X = [pt[0] for pt in points_on_surface]
    Y = [pt[1] for pt in points_on_surface]

    #for pt in points_on_surface:
        #plt.scatter(pt[0], pt[1])
    plt.axis('equal')

    weights = [1 / np.e**(i**3/100) for i in range(len(Y))]
    #weights = [1] * len(Y)
    print("Веса точек для регрессии:", weights)

    model = np.poly1d(np.polyfit(X, Y, 3, w=weights))
    #print(model)

    derivative = np.polyder(model, 1)
    tan = derivative[0]
    wheel_angle = np.pi/2 - np.arctan(tan)
    print("Полином сплайна:", model, sep='\n')
    print("Угол поворота передней тележки: ", wheel_angle / np.pi * 180, "градусов")

    polyline = np.linspace(0, 2, 50)
    plt.scatter(X, Y)
    plt.plot(polyline, model(polyline))
    plt.show()

    polyline_3d = np.array([(pt, model(pt), 0) for pt in polyline])
    for pt_3d in polyline_3d:
        pt_2d = pyar_camera.project_point(pt_3d)
        #pt_2d = pyar_camera.point_cam_to_vr(pyar_camera.project_point(pt_3d))
        pt = (int(pt_2d.x), int(pt_2d.y))
        cv2.circle(image, pt, 2, (255, 0, 0), 2)


    cv2.imshow('result', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return wheel_angle



img = cv2.imread("../img.png")
img1 = cv2.imread("../img1.png")
img1 = cv2.imread("../img1.png")
img2 = cv2.imread("../img2.png")
img3 = cv2.imread("../img3.png")
img5 = cv2.imread("../img5.png")

#angle = getWheelAngle(img)
#angle1 = getWheelAngle(img1)
#angle2 = getWheelAngle(img2)
#angle3 = getWheelAngle(img3)
angle5 = getWheelAngle(img5)

print(angle5)