from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np


# Utility class which defines the used static methods


def binarize(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    print(f"The threshold calculated by Otsu's algorithm is: {ret}")
    image_show(thresh, "Binarized image with Otsu's algorithm")
    return thresh


def connected_comp_labelling(image):
    # this uses 8-way connectivity by default
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8)
    blob_stats = blob_stats_parser(retval, stats)
    mask = blobs_mask(retval, labels.copy())
    #draw_obb(image, blob_stats)

    # print(f"num labels {retval} while labels_im {labels.shape}, {stats}")
    # labels[(labels == 0)] = 200
    # image_show(labels, "bho")


# parser used to acquire the stats of each component.
# the list contains a dictionary with all the stats for each component
def blob_stats_parser(num_labels, stats):
    blob_stats = []
    for i in range(num_labels):
        blob_stats.append({'X': stats[i, cv2.CC_STAT_LEFT],
                           'Y': stats[i, cv2.CC_STAT_TOP],
                           'W': stats[i, cv2.CC_STAT_WIDTH],
                           'H': stats[i, cv2.CC_STAT_HEIGHT],
                           'A': stats[i, cv2.CC_STAT_AREA]})
    return blob_stats


# in order to better outline the components a mask is defined
# it permits to tell apart different components using RGB colors
def blobs_mask(num_labels, labels):
    mask_rgb = cv2.merge([labels, labels, labels])
    for i in range(num_labels):
        mask_rgb[labels == i] = (np.random.choice(range(255), size=3))
    image_show(mask_rgb, "asda")
    return mask_rgb


def draw_obb(image, stats_list):
    for i in range(len(stats_list)):
        image_obb = cv2.rectangle(image, (stats_list[i]["X"], stats_list[i]["Y"]),
                                  (stats_list[i]["X"] + stats_list[i]["W"], stats_list[i]["Y"] + stats_list[i]["H"]),
                                  (0, 255, 0), 3)
        plt.imshow(image_obb)
    plt.show()


def image_show(image, title):
    plt.title(title)
    plt.imshow(image, cmap='gray', vmin='0', vmax='255')
    plt.show()
