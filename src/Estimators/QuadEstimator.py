import numpy as np

from src.Estimators.SplineEstimator import SplineEstimator


class QuadEstimator(SplineEstimator):

    def get_spline(self, surface_projections):
        X = [pt.x for pt in surface_projections]
        Y = [pt.y for pt in surface_projections]

        polynom = np.polyfit(X, Y, 4, full=True)
        return np.poly1d(polynom[0])

    def get_points(self, surface_projections):
        spline = self.get_spline(surface_projections)
        X = np.arange(-10, 10, 0.05)
        Y = spline(X)

        return [(X[i], Y[i]) for i in range(len(X))]

    def get_error(self, surface_projections):
        X = [pt.x for pt in surface_projections]
        Y = [pt.y for pt in surface_projections]

        # weights = [1.0] + [0.8] * (len(Y) - 1)
        # weights = [1 / np.e ** (i**3 / 100) for i in range(len(Y))]
        # print("Веса точек для регрессии:", weights)

        # словарь {ошибка: полином}
        # hypotheses = {
        #     biquad_polynom[1][0]: biquad_polynom[0],
        #     qubic_polynom[1][0]: qubic_polynom[0]
        # }
        # keys = list(hypotheses.keys())
        # return min(keys)

        polynom = np.polyfit(X, Y, 4, full=True)
        return polynom[1][0]


    def getWheelAngle(self, surface_projections) -> float:
        spline = self.get_spline(surface_projections)
        print("Полином сплайна:", spline, sep='\n')

        derivative = np.polyder(spline)
        tan = derivative[0]
        if tan >= 0:
            wheel_angle = np.pi / 2 - np.arctan(tan)
        else:
            wheel_angle = -np.pi / 2 - np.arctan(tan)

        return wheel_angle
