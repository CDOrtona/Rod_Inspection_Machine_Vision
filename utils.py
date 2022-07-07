import math
from turtle import Vec2D, shape
from cv2 import arcLength, cv2
from matplotlib import pyplot as plt
import numpy as np
from regex import V1
import rod

TYPE_OF_ROD = {1:"A", 2:"B"}

# Utility class which defines the used static methods
RGB_LABELS = [(0, 0, 255), (255, 127, 255), (127, 0, 255), (127, 0, 127), (0, 255, 0), (255, 255, 0),
              (0, 255, 255), (255, 0, 255), ]

rod_list = []
THRESHOLD_CIRCULARITY = 1.16

def binarize(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # ret_inv, thresh_inv = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_show(thresh, "Binarized image with Otsu's algorithm")
    # image_show(thresh_inv, "Binarized image with Otsu's algorithm")
    return thresh


# beautify
def med_filter(image, iterations):

    for i in range(iterations):
        image = cv2.medianBlur(image, 3)
    return image


def erosion(bin_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(bin_image, kernel)
    return eroded_image


def dilation(bin_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_image = cv2.dilate(bin_image, kernel)
    return dilated_image


def barycenter_distance(contour, minor_axis):
    d1 = (0, 0, 255)

    for i in range(len(minor_axis)):
        for k in range(len(contour[0])):
            dist_minor_axis = (minor_axis[i][0] * contour[0][k] + minor_axis[i][1] * contour[1][k] + minor_axis[i][2]) \
                             / math.sqrt(minor_axis[i][0] ** 2 + minor_axis[i][1] ** 2)
            if dist_minor_axis < d1[2]:
                d1 = (contour[0], contour[1], dist_minor_axis)
        distance_b = 2*d1[2]
        rod_list[i].width_b = distance_b


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
        # 
        # 
        component = np.array([[255 if pixel == i else 0 for pixel in row] for row in labels], dtype=np.uint8)
        
        if is_rod(component):
            rod_list[-1].barycenter = centroids[i]
            components_list.append(component)

        
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
    ji = np.zeros_like(components)

    for i in range(len(moments_list)):
        # orientation DEFINED IN [0, PI], FIND A WAY TO EXPRESS IT IN [0, 2PI]
        theta = -0.5 * math.atan((2 * moments_list[i]["mu11"]) / (moments_list[i]["mu02"] - moments_list[i]["mu20"]))
        d2theta = 2 * (moments_list[i]['mu02'] - moments_list[i]['mu20']) * math.cos(2 * theta) - \
                  4 * moments_list[i]['mu11'] * math.sin(2 * theta)
        theta = theta if d2theta > 0 else theta + math.pi / 2
        rod_list[i].orientation = theta * 180 / math.pi
        print(theta)

        alpha = -math.sin(theta)
        beta = math.cos(theta)

        # major axis equation in the image reference frame: aj+bi+c=0 -> a = alpha, b = -beta, c = beta*ib - alpha*jb
        major_axis.append((alpha, -beta, beta * rod_list[i].barycenter[0] - alpha * rod_list[i].barycenter[1]))
        # minor axis equation in the image reference frame: aj+bi+c=0 -> a = beta, b = alpha, c = -beta*jb - alpha*ib
        minor_axis.append((beta, alpha, -beta * rod_list[i].barycenter[1] - alpha * rod_list[i].barycenter[0]))

        # according to what do I choose the kernel size???????
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        component_eroded = cv2.erode(components[i], kernel)
        component_contour = components[i] - component_eroded
        image_show(component_contour, "Contour extrapolated from erosion")

        # furthest points from the major and minor axis
        # cx = (j, i, distance from major axis)
        c1 = (0, 0, 0)
        c2 = (0, 0, 0)
        c3 = (0, 0, 0)
        c4 = (0, 0, 0)

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

        # line connecting the farthest points from the minor and major axis -> l1 = (m, q)
        cl1 = -(alpha*c1[0] - beta*c1[1])
        cl2 = -(alpha*c2[0] - beta*c2[1])
        cw1 = -(beta*c3[0] + alpha*c3[1])
        cw2 =  -(beta*c4[0] + alpha*c4[1])

        # vertexes of the minimum enclosed rectangle
        v1 = ( (beta*cl1-alpha*cw1)/(alpha**2+beta**2) , (-beta*cw1-alpha*cl1)/(alpha**2+beta**2))
        v2 = ( (beta*cl1-alpha*cw2)/(alpha**2+beta**2) , (-beta*cw2-alpha*cl1)/(alpha**2+beta**2))
        v3 = ( (beta*cl2-alpha*cw1)/(alpha**2+beta**2) , (-beta*cw1-alpha*cl2)/(alpha**2+beta**2))
        v4 = ( (beta*cl2-alpha*cw2)/(alpha**2+beta**2) , (-beta*cw2-alpha*cl2)/(alpha**2+beta**2))

        component_rgb = cv2.merge([components[i], components[i], components[i]])
        cv2.line(component_rgb, (int(v1[0]), int(v1[1])), (int(v3[0]), int(v3[1])), (255, 0, 0), 1)
        cv2.line(component_rgb, (int(v3[0]), int(v3[1])), (int(v4[0]), int(v4[1])), (255, 0, 0), 1)
        cv2.line(component_rgb, (int(v4[0]), int(v4[1])), (int(v2[0]), int(v2[1])), (255, 0, 0), 1)
        cv2.line(component_rgb, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), (255, 0, 0), 1)
        # image_show(component_rgb, "MER")

        # component_rgb = cv2.merge([components[i], components[i], components[i]])
        cv2.circle(component_rgb, (c1[1], c1[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (c2[1], c2[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (c3[1], c3[0]), 4, (0, 150, 150), -1)
        cv2.circle(component_rgb, (c4[1], c4[0]), 4, (0, 150, 150), -1)
        leng = math.sqrt((v3[0]-v4[0])**2 + (v3[1]-v4[1])**2)
        wid = math.sqrt((v2[0]-v4[0])**2 + (v2[1]-v4[1])**2)
        image_show_LW(component_rgb, "MER", leng, wid)

    draw_major_axis(major_axis, image)
    barycenter_distance(ji, minor_axis)


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
    intersection_points = []
    comp_major_axis = np.zeros_like(image)
    for i in range(len(major_axis)):
        
        intersection_points.append((0,-major_axis[i][2]/major_axis[i][0]))
        intersection_points.append((np.shape(image)[1], -(np.shape(image)[1]*major_axis[i][1]+major_axis[i][2])/major_axis[i][0]))
        intersection_points.append(( -major_axis[i][2]/major_axis[i][1],0))
        intersection_points.append((-(major_axis[i][0]*np.shape(image)[0]+major_axis[i][2])/major_axis[i][1],np.shape(image)[0]))

        for point in intersection_points:
            if(point[0]<0 or point[0]>np.shape(image)[1] or point[1]<0 or point[1]>np.shape(image)[0]):
                intersection_points.remove(point)


        comp_major_axis = cv2.line(image, (int(intersection_points[-2][0]), int(intersection_points[-2][1])), (int(intersection_points[-1][0]), int(intersection_points[-1][1])), (255, 0, 0), 1)

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


def image_show_LW(image, title, length, width):
    font2 = {'family':'serif','color':'darkred','size':15}
    plt.title(title)
    plt.imshow(image, cmap='gray', vmin='0', vmax='255')
    plt.xlabel("Lenght : " + str(length) + "  Width : " + str(width), fontdict = font2)
    plt.show()

def is_rod(component):
    
    contours, hierarchy = cv2.findContours(component, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print(np.shape(hierarchy))

    if np.shape(hierarchy)[1] == 1 :
        return False

    if np.shape(hierarchy)[1] != 1 :

        perimeter = cv2.arcLength(contours[-1], True )
        area = cv2.contourArea(contours[-1], False)
        circularity = (perimeter**2)/(area*4*math.pi)

        if circularity < THRESHOLD_CIRCULARITY : 
            return False
            
        rod_list.append(rod.Rod())
        
        number_of_holes = np.shape(hierarchy)[1]-1
        rod_list[-1].type = TYPE_OF_ROD[number_of_holes] 


        for i in range(number_of_holes):
            moment = cv2.moments(contours[i])
            circle_barycenter = ((moment["m10"]/moment["m00"]),(moment["m01"]/moment["m00"])) 
            diameter = cv2.arcLength(contours[i], True)/math.pi
            rod_list[-1].assign_holes(diameter, circle_barycenter)

        return True      

