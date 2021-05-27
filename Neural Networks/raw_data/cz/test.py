import cv2
import matplotlib.pyplot as plt


image = cv2.imread("heights.pgm", cv2.IMREAD_GRAYSCALE)
plt.imshow(image)
plt.show()