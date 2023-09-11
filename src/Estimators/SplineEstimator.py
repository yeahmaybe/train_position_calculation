from abc import abstractmethod, ABC


class SplineEstimator(ABC):
    @abstractmethod
    def get_error(self, surface_projections) -> float:
        pass

    @abstractmethod
    def getWheelAngle(self, surface_projections) -> float:
        pass