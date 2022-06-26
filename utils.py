from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np

# Utility class which defines the used static methods


def binarize(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(f"The threshold calculated by Otsu's algorithm is: {ret}")
    image_show(thresh, "Binarized image with Otsu's algorithm")
    return thresh


def connected_comp_labelling(image):
    # this uses 8-way connectivity by default
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 4)
    blob_stats = blob_stats_parser(retval, stats)
    # print(f"num labels {retval} while labels_im {labels.shape}, {stats}")
    # labels[(labels == 0)] = 200
    # image_show(labels, "bho")


def blob_stats_parser(num_labels, stats):
    blob_stats = []
    for i in range(num_labels):
        blob_stats.append({'X': stats[i, cv2.CC_STAT_LEFT],
                           'Y' : stats[i, cv2.CC_STAT_TOP],
                           'W' : stats[i, cv2.CC_STAT_WIDTH],
                           'H' : stats[i, cv2.CC_STAT_HEIGHT],
                           'A' : stats[i, cv2.CC_STAT_AREA]})
    return blob_stats

def image_show(image, title):
    plt.title(title)
    plt.imshow(image, cmap='gray', vmin='0', vmax='255')
    plt.show()


