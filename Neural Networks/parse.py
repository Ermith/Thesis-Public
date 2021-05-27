from libraries.parsing import parse
from libraries.matrix_manipulation import derivatives, rescale
import cv2
import os

def parse_image(image_name, output_folder_name, iterations=4, blur=False, compute_differences=False):
    cut = 5

    if not os.path.exists(output_folder_name):
	    os.makedirs(output_folder_name)

    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = rescale(image)

    if (blur):
        image = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)

        for i in range(iterations):
            reduction = 4**i
            parse(image, cut, reduction, f"{output_folder_name}/layer_blurry_{reduction}x")
        
    elif (compute_differences):
        ders_r, ders_c = derivatives(image)

        for i in range(iterations):
            reduction = 4**i
            parse(ders_r, cut, reduction, f"{output_folder_name}/differences_rows_layer_{reduction}x")
            parse(ders_c, cut, reduction, f"{output_folder_name}/differences_columns_layer_{reduction}x")
    else:
        for i in range(iterations):
            reduction = 4**i
            parse(image, cut, reduction, f"{output_folder_name}/layer_{reduction}x")