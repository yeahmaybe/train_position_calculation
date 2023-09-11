import cv2
import matplotlib.pyplot as plt

from src.AngleCalculation.AngleCalculator import AngleCalculator
from src.Image.utils import save_as_gif, get_images


def calculate_angle_and_make_gif():
    images = get_images('../TramPathIns', 126, 136, [105, 111])

    angle_calculator = AngleCalculator(images)
    angle_calculator.process_images()

    processed_images = angle_calculator.get_processed_images()
    save_as_gif(processed_images, 'frames.gif', 500)

    angles = angle_calculator.get_angles()
    plt.plot(range(len(angles)), angles)
    plt.show()


# main function

calculate_angle_and_make_gif()
cv2.destroyAllWindows()
cv2.waitKey()
