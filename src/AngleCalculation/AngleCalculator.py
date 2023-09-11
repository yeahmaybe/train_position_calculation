import numpy as np

from src.Estimators import EstimatorManager
from src.Filter.SimpleFilter import SimpleFilter
from src.Geometry.ProjectionsManager import ProjectionsManager
from src.Image.utils import show_bird_eye_view, show_process_result


class AngleCalculator:
    __images = []
    __processed_images = []
    __angles = []
    __prev_angle = None

    def __init__(self, images):
        self.__images = images

    def process_images(self):
        projections_manager = ProjectionsManager('../data/city/leftImage2.yml')

        for image in self.__images:

            try:
                surface_projections = projections_manager.get_surface_projections(image)
                print("Точки:", surface_projections)

                projections_manager.draw_points3d(surface_projections, image)

                estimator = EstimatorManager.choose_estimator(surface_projections, self.__prev_angle)
                wheel_angle = estimator.getWheelAngle(surface_projections) / np.pi * 180

                angle_filter = SimpleFilter()
                wheel_angle = angle_filter.filtrate(wheel_angle, self.__prev_angle)
                print(self.__prev_angle, wheel_angle)

                self.__prev_angle = wheel_angle
                self.__angles.append(wheel_angle)
                self.__processed_images.append(image)

                print("Угол поворота передней тележки: ", wheel_angle, "градусов")
                print("==============================")

                show_process_result(image, wheel_angle)
                show_bird_eye_view(surface_projections)

                continue

            finally:
                self.__angles.append(self.__prev_angle)

    def get_angles(self):
        return self.__angles

    def get_processed_images(self):
        return self.__processed_images
