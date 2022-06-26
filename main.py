from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np
import utils


def init():
    # Add multiple file management
    im = cv2.imread('images/TESI00.BMP', cv2.IMREAD_GRAYSCALE)
    hist, bins = np.histogram(im.flatten(), 256, [0, 256])
    # plt.figure(1)
    # plt.stem(hist, use_line_collection=True)
    # plt.figure(2)
    # plt.imshow(im, cmap='gray', vmin='0', vmax='255')
    # plt.show()
    return im


if __name__ == '__main__':
    image = init()
    bin_image = utils.binarize(image.copy())
    utils.connected_comp_labelling(bin_image)



