from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np
import utils


def init():
    # Add multiple file management
    im = cv2.imread('images/TESI01.BMP')
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # hist, bins = np.histogram(im.flatten(), 256, [0, 256])
    # plt.figure(1)
    # plt.stem(hist, use_line_collection=True)
    # plt.figure(2)
    # plt.imshow(im, cmap='gray', vmin='0', vmax='255')
    # plt.show()
    return im_rgb, im_gray


if __name__ == '__main__':
    image_rgb, image_gray = init()
    utils.image_show(image_gray, "main")
    bin_image = utils.binarize(image_gray.copy())
    # clean_image = utils.bin_morphology(bin_image)
    components, rods = utils.connected_comp_labelling(bin_image)
    utils.blob_analysis(components, image_rgb.copy())
#    print(rod_list[1].holes)
    print(list(rods[i] for i in range(len(rods))))




