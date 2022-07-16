import glob

from cv2 import cv2
import utils


if __name__ == '__main__':
    images = [cv2.imread(image) for image in glob.glob('images/*.BMP')]
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        utils.image_show(image_gray, "Input Image")
        filtered_image = utils.med_filter(image_gray, 4)
        # utils.image_show(filtered_image, "Filtered Image")
        bin_image = utils.binarize(filtered_image.copy())
        utils.detach_components(bin_image)
        # eroded_image = utils.erosion(bin_image)
        # utils.image_show(eroded_image, "Eroded Image")
        components, rods = utils.connected_comp_labelling(bin_image)
        utils.blob_analysis(components, image_rgb.copy())
        print(list(rods[i] for i in range(len(rods))))





