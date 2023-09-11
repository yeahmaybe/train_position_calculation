from src.Estimators.CircleEstimator import CircleEstimator
from src.Estimators.LineEstimator import LineEstimator
from src.Estimators.QuadEstimator import QuadEstimator
from src.Estimators.QubEstimator import QubEstimator


def choose_estimator(surface_projections, prev_angle):
    """
    Choose a spline function with the lowest MSE

    :param surface_projections: known points on a rails line
    :param prev_angle: previous wheel angle
    :return:
    """
    circle_error = CircleEstimator().get_error(surface_projections)
    line_error = LineEstimator().get_error(surface_projections)
    polynome_error = QuadEstimator().get_error(surface_projections)
    poly_estimator = QuadEstimator()
    if prev_angle is None or abs(prev_angle) > 2:
        polynome_error = QubEstimator().get_error(surface_projections)
        poly_estimator = QubEstimator()

    hypotheses = {
        circle_error: CircleEstimator(),
        polynome_error: poly_estimator,
        line_error: LineEstimator()
    }
    least_error = min(hypotheses.keys())

    print("Ошибка на окружности: ", circle_error)
    print("Ошибка на полиномах: ", polynome_error)
    print("Ошибка на прямой: ", line_error)
    return hypotheses[least_error]
