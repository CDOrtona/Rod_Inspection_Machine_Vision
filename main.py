from cv2 import cv2
import utils
import pandas as pd


def export_cvs(rods_list):
    data = []
    for i in range(len(rods_list)):
        data.append(
            [rods_list[i].type, [rods_list[i].barycenter[0], rods_list[i].barycenter[0]], rods_list[i].orientation,
             rods_list[i].length, rods_list[i].width, rods_list[i].width_b, rods_list[i].holes]
        )

    df = pd.DataFrame(data, columns=['Type', 'Barycenter Coordinates', 'Orientation', 'Length', 'Width', 'Width_b',
                                     'Holes: Diameter, Barycenter coordinates'])
    # Excel file must be closed before running the script
    df.to_csv('ANALYSIS_RESULTS.csv')


if __name__ == '__main__':
    image = cv2.imread('images/TESI51.BMP')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    utils.image_show(image_gray, "Input Image")
    filtered_image = utils.med_filter(image_gray, 4)
    bin_image = utils.binarize(filtered_image.copy())
    utils.detach_components(bin_image)
    components, rods = utils.connected_comp_labelling(bin_image)
    utils.blob_analysis(components, image_gray.copy())
    print(list(rods[i] for i in range(len(rods))))
    export_cvs(rods)
