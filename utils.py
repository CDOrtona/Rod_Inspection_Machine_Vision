import math
from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np
import rod

TYPE_OF_ROD = {1: "A", 2: "B"}

RGB_LABELS = [(0, 0, 255), (255, 127, 255), (127, 0, 255), (127, 0, 127), (0, 255, 0), (255, 255, 0),
              (0, 255, 255), (255, 0, 255), ]

rod_list = []
components_list = []
THRESHOLD_CIRCULARITY = 1.16
THRESHOLD_AREA = 6000


# function used to binarize image
def binarize(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    image_show(thresh, "Binarized image with Otsu's algorithm")
    return thresh


# median filter used to remove salt and pepper noise(iron powder)
def med_filter(image, iterations):
    for i in range(iterations):
        image = cv2.medianBlur(image, 3)
    return image


# function used to computer the erosion of each component
def erosion(bin_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(bin_image, kernel)
    return eroded_image


# function used to detach touching rods
def detach_components(bin_image):
    while True:

        contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        exit_form_while = True
        idx = 0
        i = 0

        # check for each component the area, if it's greater than 6000 then it's two or more rods touching
        for cnt in contours:

            if abs(cv2.contourArea(cnt, True)) > 6000:
                exit_form_while = False
                idx = i

            i = i + 1
        # condition when no touching components are detected, hence we exit while
        if exit_form_while:
            break

        # we assign the external contour of object detected to be formed by multiple rods
        contour = contours[idx]
        # convex hull and convexity defect are computed accordingly
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # contour's channels are indexed by the index found with convexityDefects
        far_list = []
        for i in range(np.shape(defects)[0]):
            _, _, f, d = defects[i, 0]
            # far is the pair of coordinates identifying the point where a concavity occurs
            far = tuple(contour[f][0])

            far_list.append([far, d / 256.0])

        far_list = sorted(far_list, key=lambda x: x[1])

        start_far = far_list[-1][0]
        end_far = far_list[-2][0]
        dist_far = math.sqrt((start_far[0] - end_far[0]) ** 2 + (start_far[1] - end_far[1]) ** 2)

        # the distance between each of the two points belonging to the last four elements of the list are checked to
        # be greater than 30, if the condition is fulfilled then we switch to another pair of points
        if dist_far > 30:
            end_far = far_list[-3][0]
            dist_far = math.sqrt((start_far[0] - end_far[0]) ** 2 + (start_far[1] - end_far[1]) ** 2)
            if dist_far > 30:
                end_far = far_list[-4][0]

        # A line is drawn in order to detach the rods
        cv2.line(bin_image, start_far, end_far, (0, 0, 0), 2)
        image_show(bin_image, "Approximated Contours")


def connected_comp_labelling(image):
    # function used to detect the different connected components in our image
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 4)

    # for loop starts from 1 since the first blob is always the background
    for i in range(1, retval):
        # each integer of the component is given the value 255, hence labeled as foreground
        component = np.array([[255 if pixel == i else 0 for pixel in row] for row in labels], dtype=np.uint8)

        # each found component is checked in order to asses whether it is a rod or not
        if is_rod(component):
            rod_list[-1].barycenter = centroids[i]
            components_list.append(component)

    blobs_mask(retval, labels.copy())

    return components_list, rod_list


def blob_analysis(components, image):
    moments_list = [cv2.moments(components[i]) for i in range(len(components))]

    major_axis = []
    minor_axis = []

    for i in range(len(moments_list)):
        theta = -0.5 * math.atan((2 * moments_list[i]["mu11"]) / (moments_list[i]["mu02"] - moments_list[i]["mu20"]))
        d2theta = 2 * (moments_list[i]['mu02'] - moments_list[i]['mu20']) * math.cos(2 * theta) - \
                  4 * moments_list[i]['mu11'] * math.sin(2 * theta)
        theta = theta if d2theta > 0 else theta + math.pi / 2
        rod_list[i].orientation = theta * 180 / math.pi

        alpha = -math.sin(theta)
        beta = math.cos(theta)

        # major axis equation in the image reference frame: aj+bi+c=0 -> a = alpha, b = -beta, c = beta*ib - alpha*jb
        major_axis.append((alpha, -beta, beta * rod_list[i].barycenter[0] - alpha * rod_list[i].barycenter[1]))
        # minor axis equation in the image reference frame: aj+bi+c=0 -> a = beta, b = alpha, c = -beta*jb - alpha*ib
        minor_axis.append((beta, alpha, -beta * rod_list[i].barycenter[1] - alpha * rod_list[i].barycenter[0]))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        component_eroded = cv2.erode(components[i], kernel)
        # we extrapolate the contour by dividing the image by its eroded pair
        component_contour = components[i] - component_eroded
        image_show(component_contour, "Contour extrapolated from erosion")

        # furthest points from the major and minor axis
        # cx = (j, i, distance from major axis)
        c1 = (0, 0, 0)
        c2 = (0, 0, 0)
        c3 = (0, 0, 0)
        c4 = (0, 0, 0)

        # points which have the minimum distance from the minor axis, they can either be on the left or right
        # of the major axis depending on weather the distance is positive or negative
        wb_1 = (0, 0, 255)
        wb_2 = (0, 0, 255)

        # the distance between each point belonging to the contour and the major and minor axis is calculated in order
        # to find those points c1, c2, c3, c4 which are the furthest from the major axis and minor axis
        ji = np.nonzero(component_contour == 255)  # tuple of 2 ndarrays
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

            # the two points the closest to the minor axis are calculated in order to compute the width at the
            # barycenter
            if abs(dist_minor_axis) < wb_1[2] and dist_major_axis > 0:
                wb_1 = (ji[0][k], ji[1][k], abs(dist_minor_axis))
            if abs(dist_minor_axis) < wb_2[2] and dist_major_axis < 0:
                wb_2 = (ji[0][k], ji[1][k], abs(dist_minor_axis))

        # width at the barycenter
        width_b = math.sqrt((wb_1[0] - wb_2[0]) ** 2 + (wb_1[1] - wb_2[1]) ** 2)
        rod_list[i].width_b = width_b

        # line connecting the farthest points from the minor and major axis -> l1 = (m, q)
        cl1 = -(alpha * c1[0] - beta * c1[1])
        cl2 = -(alpha * c2[0] - beta * c2[1])
        cw1 = -(beta * c3[0] + alpha * c3[1])
        cw2 = -(beta * c4[0] + alpha * c4[1])

        # vertexes of the minimum enclosed rectangle
        v1 = (
        (beta * cl1 - alpha * cw1) / (alpha ** 2 + beta ** 2), (-beta * cw1 - alpha * cl1) / (alpha ** 2 + beta ** 2))
        v2 = (
        (beta * cl1 - alpha * cw2) / (alpha ** 2 + beta ** 2), (-beta * cw2 - alpha * cl1) / (alpha ** 2 + beta ** 2))
        v3 = (
        (beta * cl2 - alpha * cw1) / (alpha ** 2 + beta ** 2), (-beta * cw1 - alpha * cl2) / (alpha ** 2 + beta ** 2))
        v4 = (
        (beta * cl2 - alpha * cw2) / (alpha ** 2 + beta ** 2), (-beta * cw2 - alpha * cl2) / (alpha ** 2 + beta ** 2))

        component_rgb = cv2.merge([components[i], components[i], components[i]])
        cv2.line(component_rgb, (int(v1[0]), int(v1[1])), (int(v3[0]), int(v3[1])), (255, 0, 0), 1)
        cv2.line(component_rgb, (int(v3[0]), int(v3[1])), (int(v4[0]), int(v4[1])), (255, 0, 0), 1)
        cv2.line(component_rgb, (int(v4[0]), int(v4[1])), (int(v2[0]), int(v2[1])), (255, 0, 0), 1)
        cv2.line(component_rgb, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), (255, 0, 0), 1)

        cv2.circle(component_rgb, (c1[1], c1[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (c2[1], c2[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (c3[1], c3[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (c4[1], c4[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (wb_1[1], wb_1[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (wb_2[1], wb_2[0]), 4, (0, 150, 150), -1)
        plt.annotate("wb_1",  # this is the text
                     (wb_1[1], wb_1[0]),  # these are the coordinates to position the label
                     textcoords="offset points", size=16, color= "green",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     ha='center')
        plt.annotate("wb_2",  # this is the text
                     (wb_2[1], wb_2[0]),  # these are the coordinates to position the label
                     textcoords="offset points", size=16, color="green",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     ha='center')

        # length and width are computed from the MER
        length = math.sqrt((v3[0] - v4[0]) ** 2 + (v3[1] - v4[1]) ** 2)
        width = math.sqrt((v2[0] - v4[0]) ** 2 + (v2[1] - v4[1]) ** 2)
        rod_list[i].length = length
        rod_list[i].width = width
        image_show_LW(component_rgb, "MER", length, width)

    draw_major_axis(major_axis, image)


# in order to better outline the components a mask is defined
# it permits to tell apart different components using RGB colors
def blobs_mask(num_labels, labels):
    mask_rgb = cv2.merge([labels, labels, labels])
    for i in range(1, num_labels):
        mask_rgb[labels == i] = RGB_LABELS[i]
    image_show(mask_rgb, "Outlined Blobs")
    return mask_rgb


def draw_major_axis(major_axis, image):
    image = cv2.merge([image, image, image])
    intersection_points = []
    comp_major_axis = np.zeros_like(image)
    for i in range(len(major_axis)):

        intersection_points.append((0, -major_axis[i][2] / major_axis[i][0]))
        intersection_points.append(
            (np.shape(image)[1], -(np.shape(image)[1] * major_axis[i][1] + major_axis[i][2]) / major_axis[i][0]))
        intersection_points.append((-major_axis[i][2] / major_axis[i][1], 0))
        intersection_points.append(
            (-(major_axis[i][0] * np.shape(image)[0] + major_axis[i][2]) / major_axis[i][1], np.shape(image)[0]))
        static_leng = len(intersection_points)

        for i in range(static_leng):
            point = intersection_points[static_leng - 1 - i]
            if point[0] < 0 or point[0] > np.shape(image)[1] or point[1] < 0 or point[1] > np.shape(image)[0]:
                intersection_points.remove(point)

        comp_major_axis = cv2.line(image, (int(intersection_points[-2][0]), int(intersection_points[-2][1])),
                                   (int(intersection_points[-1][0]), int(intersection_points[-1][1])), (255, 0, 0), 1)
    image_show(comp_major_axis, "Image with Major Axis")


def image_show(image, title):
    plt.title(title)
    plt.imshow(image, cmap='gray', vmin='0', vmax='255')
    plt.show()


def image_show_LW(image, title, length, width):
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
    plt.title(title)
    plt.imshow(image, cmap='gray', vmin='0', vmax='255')
    plt.xlabel("Lenght : " + str(length) + "  Width : " + str(width), fontdict=font2)
    plt.show()


# function used to filter out distractors
def is_rod(component):
    contours, hierarchy = cv2.findContours(component, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # screw filtering
    if np.shape(hierarchy)[1] == 1:
        return False

    # washers filtering
    if np.shape(hierarchy)[1] != 1:

        perimeter = cv2.arcLength(contours[-1], True)
        area = cv2.contourArea(contours[-1], False)
        circularity = (perimeter ** 2) / (area * 4 * math.pi)

        if circularity < THRESHOLD_CIRCULARITY:
            return False

        rod_list.append(rod.Rod())

        number_of_holes = np.shape(hierarchy)[1] - 1
        rod_list[-1].type = TYPE_OF_ROD[number_of_holes]

        # for each hole its diameter and barycenter is computed
        for i in range(number_of_holes):
            moment = cv2.moments(contours[i])
            circle_barycenter = ((moment["m10"] / moment["m00"]), (moment["m01"] / moment["m00"]))
            diameter = cv2.arcLength(contours[i], True) / math.pi
            rod_list[-1].assign_holes(diameter, circle_barycenter)

        return True
