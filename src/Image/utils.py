import cv2
import imageio
from matplotlib import pyplot as plt

from pyar import Point3D


def get_img_name(i: int):
    img_name = str(i)
    while len(img_name) < 4:
        img_name = '0' + img_name
    return img_name


def get_images(folder_path, start_idx, finish_idx, skip=None):
    if skip is None:
        skip = []

    result = []
    for i in range(start_idx, finish_idx + 1):
        if i in skip:
            continue

        name = get_img_name(i)
        path = "{0}/{1}.png".format(folder_path, name)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result.append(img)
    return result


def save_as_gif(images, name, duration) -> None:
    imageio.mimsave(name,  # output gif
                    images,  # array of input frames
                    duration=duration)


def show_bird_eye_view(points: list[Point3D]) -> None:
    """
    Shows a plot of given points in a Z=const plain

    :param points: list of points
    :return:
    """
    X = [pt.x for pt in points]
    Y = [pt.y for pt in points]
    plt.scatter(X, Y)
    plt.axis('equal')
    ax = plt.gca()
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.show()


def show_process_result(image, wheel_angle):
    cv2.putText(image, "ANGLE: " + str(wheel_angle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 150), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('result', image)
    cv2.waitKey()
