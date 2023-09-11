import cv2

from src.AxislineDetection import AxislineDetector
from pyar.camera import Camera as pyarCamera


class ProjectionsManager:
    def __init__(self, file_name):
        self.__pyar_camera = pyarCamera.from_yaml(file_name)

    def draw_points3d(self, surface_projections, image):
        projections_2d = list(map(self.__pyar_camera.project_point, surface_projections))
        for p in projections_2d:
            cv2.circle(image, (int(p.x), int(p.y)), 5, (20, 20, 255), 2)

    def get_surface_projections(self, image):
        return AxislineDetector.get_surface_projections(self.__pyar_camera, image)
