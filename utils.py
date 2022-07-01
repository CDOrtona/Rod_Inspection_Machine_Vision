import math
from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np
import rod

# Utility class which defines the used static methods
RGB_LABELS = [(0, 0, 255), (255, 127, 255), (127, 0, 255), (127, 0, 127), (0, 255, 0), (255, 255, 0),
              (0, 255, 255), (255, 0, 255), ]

rod_list = []


def binarize(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # ret_inv, thresh_inv = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_show(thresh, "Binarized image with Otsu's algorithm")
    # image_show(thresh_inv, "Binarized image with Otsu's algorithm")
    return thresh


# def bin_morphology(image):
#     # kernel = np.eye(5, dtype=np.uint8)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cleaned_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
#     image_show(cleaned_image, "After Morphology")
#     return cleaned_image


def connected_comp_labelling(image):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8)
    rod_stats = blob_stats_parser(retval, stats, centroids)
    print(rod_stats)

    components_list = []

    # for loop starts from 1 since the first blob is always the background
    for i in range(1, retval):
        # I'll have to add a checker for checking whether it's a rod or not
        rod_list.append(rod.Rod())
        rod_list[i - 1].barycenter = centroids[i]
        component = np.array([[255 if pixel == i else 0 for pixel in row] for row in labels], dtype=np.uint8)
        components_list.append(component)
        # image_show(255-component, "component")
        h_retval, h_labels, h_stats, h_centroids = cv2.connectedComponentsWithStats(255 - component, 8)
        rod_list[i - 1].assign_holes(h_retval, h_stats, h_centroids)
    #     mask_holes = blobs_mask(h_retval, h_labels)
    #     rod_stats = blob_stats_parser(h_retval, h_stats, h_centroids)
    #     image_mer = draw_obb(mask_holes, rod_stats)
    #     image_centroids = draw_centroids(image_mer, rod_stats)
    #     image_show(image_centroids, "qwqwqwe")
    #
    mask1 = blobs_mask(retval, labels.copy())
    image_mer1 = draw_obb(mask1, rod_stats)
    image_centroids1 = draw_centroids(image_mer1, rod_stats)
    image_show(image_centroids1, "MER")

    return components_list, rod_list


def blob_analysis(components, image):
    moments_list = [cv2.moments(components[i]) for i in range(len(components))]

    major_axis = []
    minor_axis = []

    for i in range(len(moments_list)):
        # orientation DEFINED IN [0, PI], FIND A WAY TO EXPRESS IT IN [0, 2PI]
        theta = -0.5 * math.atan(2 * moments_list[i]["mu11"] / (moments_list[i]["mu02"] - moments_list[i]["mu20"]))
        rod_list[i].orientation = abs(theta) * 180 / math.pi

        alpha = -math.sin(theta)
        beta = math.cos(theta)

        # major axis equation in the image reference frame: aj+bi+c=0 -> a = alpha, b = -beta, c = beta*ib - alpha*jb
        major_axis.append((alpha, -beta, beta * rod_list[i].barycenter[0] - alpha * rod_list[i].barycenter[1]))
        # minor axis equation in the image reference frame: aj+bi+c=0 -> a = beta, b = alpha, c = -beta*jb - alpha*ib
        minor_axis.append((beta, -alpha, -beta * rod_list[i].barycenter[1] - alpha * rod_list[i].barycenter[0]))

        # according to what do I choose the kernel size???????
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        component_eroded = cv2.erode(components[i], kernel)
        component_contour = components[i] - component_eroded
        image_show(component_contour, "Contour extrapolated from erosion")
        print(component_contour)

        # furthest points from the major and minor axis
        # cx = (j, i, distance from major axis)
        c1 = (0, 0, 0)
        c2 = (0, 0, 0)
        c3 = (0, 0, 0)
        c4 = (0, 0, 0)

        # vertex of the minimum enclosed rectangle
        v1 = (0, 0, 0)
        v2 = (0, 0, 0)
        v3 = (0, 0, 0)
        v4 = (0, 0, 0)

        ji = np.nonzero(component_contour == 255)  # tuple of 2 ndarrays
        print(ji, type(ji[0]), len(ji[1]))
        for k in range(len(ji[0])):
            dist_major_axis = (major_axis[i][0] * ji[0][k] + major_axis[i][1] * ji[1][k] + major_axis[i][2]) / \
                              math.sqrt(major_axis[i][0] ** 2 + major_axis[i][1] ** 2)
            dist_minor_axis = (minor_axis[i][0] * ji[0][k] + minor_axis[i][1] * ji[1][k] + minor_axis[i][2]) / \
                              math.sqrt(minor_axis[i][0] ** 2 + minor_axis[i][1] ** 2)
            if dist_major_axis > c1[2]:
                c1 = (ji[0][k], ji[1][k], dist_major_axis)
            if dist_major_axis < c2[2]:
                c2 = (ji[0][k], ji[1][k], dist_major_axis)
            if dist_minor_axis > c3[2]:
                c3 = (ji[0][k], ji[1][k], dist_minor_axis)
            if dist_minor_axis < c4[2]:
                c4 = (ji[0][k], ji[1][k], dist_minor_axis)

        component_rgb = cv2.merge([components[i], components[i], components[i]])
        cv2.circle(component_rgb, (c1[1], c1[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (c2[1], c2[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (c3[1], c3[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (c4[1], c4[0]), 4, (0, 150, 150), -1)
        image_show(component_rgb, "MER")

    draw_major_axis(major_axis, image)


# parser used to acquire the stats of each component.
# the list contains a dictionary with all the stats for each component
def blob_stats_parser(num_labels, stats, centroids):
    blob_stats = []
    for i in range(1, num_labels):
        blob_stats.append({'X': stats[i, cv2.CC_STAT_LEFT],
                           'Y': stats[i, cv2.CC_STAT_TOP],
                           'W': stats[i, cv2.CC_STAT_WIDTH],
                           'H': stats[i, cv2.CC_STAT_HEIGHT],
                           'A': stats[i, cv2.CC_STAT_AREA],
                           'Cx': centroids[i, 0],
                           'Cy': centroids[i, 1]})
    return blob_stats


# in order to better outline the components a mask is defined
# it permits to tell apart different components using RGB colors
def blobs_mask(num_labels, labels):
    mask_rgb = cv2.merge([labels, labels, labels])
    for i in range(1, num_labels):
        # if i != 0: # I'm not considering the background
        # mask_rgb[labels == i] = (np.random.choice(range(255), size=3))
        mask_rgb[labels == i] = RGB_LABELS[i]
    image_show(mask_rgb, "Outlined Blobs")
    return mask_rgb


# circularity feature is used to asses whether the object is a circle or not
# def is_circle(component):
#     circularity = (4 * component["A"]) / (np.pi * component["W"]**2)
#     print(f"Circularity {circularity}")
#     if 1.4 >= circularity >= .7:
#         return True
#     else:
#         return False

def draw_major_axis(major_axis, image):
    comp_major_axis = np.zeros_like(image)
    for i in range(len(major_axis)):
        x1 = -major_axis[i][2] / major_axis[i][1]
        y2 = -major_axis[i][2] / major_axis[i][0]
        comp_major_axis = cv2.line(image, (int(abs(x1)), 0), (0, int(abs(y2))), (255, 0, 0), 1)

    image_show(comp_major_axis, "Image with Major Axis")


def draw_obb(image, stats_list):
    for i in range(len(stats_list)):
        image_obb = cv2.rectangle(image, (stats_list[i]["X"], stats_list[i]["Y"]),
                                  (stats_list[i]["X"] + stats_list[i]["W"], stats_list[i]["Y"] + stats_list[i]["H"]),
                                  (255, 0, 0), 1)
        # image_centroid = cv2.line(image, stats_list[i]["Cx"], stats_list[i]["Cy"], (0, 150, 150))
        # plt.imshow(image_centroid)
    return image_obb


def draw_centroids(image, stats_list):
    for i in range(len(stats_list)):
        image_centroid = cv2.circle(image, (int(stats_list[i]["Cx"]), int(stats_list[i]["Cy"])), 4, (0, 150, 150), -1)
    return image_centroid


def image_show(image, title):
    plt.title(title)
    plt.imshow(image, cmap='gray', vmin='0', vmax='255')
    plt.show()
