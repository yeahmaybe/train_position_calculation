import numpy as np

from src.Estimators.SplineEstimator import SplineEstimator


class LineEstimator(SplineEstimator):
    b = -4.6
    def get_best_A(self, surface_projections):
        best_A = 10000

        def check_a():
            min_error = 1000000
            best_a = 1000000
            errors = []
            for pt in surface_projections:
                error = np.sqrt((a * pt.x + self.b - pt.y)**2 / (a * a + 1))
                errors.append(error)

            if min_error > np.sum(errors):
                min_error = np.sum(errors)
                best_a = a
            return best_a, min_error

        A = np.concatenate([np.arange(10, 100, 0.01), np.arange(101, 5000, 1), np.arange(-5000, -101, 1), np.arange(-100, -10, 0.01)])
        min_er = 10000000
        for a in A:
            A, er = check_a()
            if min_er > er:
                best_A = A
                min_er = er

        return best_A

    def getWheelAngle(self, surface_projections) -> float:
        tan = self.get_best_A(surface_projections)
        if tan >= 0:
            wheel_angle = np.pi / 2 - np.arctan(tan)
        else:
            wheel_angle = -np.pi / 2 - np.arctan(tan)
        return wheel_angle


    def get_points(self, surface_projections):
        best_a = self.get_best_A(surface_projections)
        X = np.arange(-10, 10, 0.01)
        points = []
        for x in X:
            points.append((x, best_a*x+self.b))

        return points

    def get_error(self, surface_projections):
        a = self.get_best_A(surface_projections)
        print("Лучший А: ", a)
        errors=[]
        for pt in surface_projections:
            error = np.sqrt((a * pt.x + self.b - pt.y)**2 / (a * a + 1))
            errors.append(error)
        print(errors)
        return np.sum(errors)