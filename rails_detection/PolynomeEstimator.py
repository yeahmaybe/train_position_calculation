import numpy as np

class PolynomeEstimator:

    def get_spline(self, surface_projections):
        X = [pt.x for pt in surface_projections]
        Y = [pt.y for pt in surface_projections]

        weights = [1.0] + [0.05] * (len(Y) - 1)
        #weights = [1 / np.e ** (i**3 / 100) for i in range(len(Y))]
        # print("Веса точек для регрессии:", weights)

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

    def get_points(self, surface_projections):
        spline = self.get_spline(surface_projections)
        X = np.arange(-10, 10, 0.05)
        Y = spline(X)

        return [(X[i], Y[i]) for i in range(len(X))]

    def get_error(self, surface_projections):
        points = self.get_points(surface_projections)
        prs_circ = []
        for pt in points:
            pt = (round(pt[0], 2), round(pt[1], 2), 0)
            prs_circ.append((pt[0], pt[1]))
        mins = []

        for pt in surface_projections:
            list_r = []
            for pt_c in prs_circ:
                list_r.append((((pt.x - pt_c[0]) ** 2) + (pt.y - pt_c[1]) ** 2) ** (1 / 2))
            mins.append(min(list_r) ** 2)
        return sum(mins)

    def getWheelAngle(self, surface_projections) -> float:
        spline = self.get_spline(surface_projections)
        print("Полином сплайна:", spline, sep='\n')

        derivative = np.polyder(spline, 1)
        tan = derivative[0]
        wheel_angle = min(np.pi / 2 - np.arctan(tan), np.pi / 2 + np.arctan(tan))

        return wheel_angle
