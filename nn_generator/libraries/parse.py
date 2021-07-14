from .basic.parsing import parse
from .basic.matrix_manipulation import derivatives, rescale
import cv2
import os

def parse_image(image_name, output_folder_name, reductions=[1, 4, 16, 64], blur=False, compute_differences=False):
    """Parses a single image into layers specifyied by reductions.
    image_name: str
        Name of the image to parse.
    output_folder_name: str
        Where to store the output.
    reductions: [int]
        Specifies into what layers is the image parsed.
    blur: bool
        Apply gaussian blur on the image before parsing.
    compute_differences: bool
        Rather then with the original image, computes 2 images containing differences in x and y direction respectively.
        Useful for heights.
    """
    cut = 5

    if not os.path.exists(output_folder_name):
	    os.makedirs(output_folder_name)

    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = rescale(image)

    if (blur):
        image = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)

        for reduction in reductions:
            parse(image, cut, reduction, f"{output_folder_name}/layer_blurry_{reduction}x")
        
    elif (compute_differences):
        ders_r, ders_c = derivatives(image)

        for reduction in reductions:
            parse(ders_r, cut, reduction, f"{output_folder_name}/differences_rows_layer_{reduction}x")
            parse(ders_c, cut, reduction, f"{output_folder_name}/differences_columns_layer_{reduction}x")
    else:
        for reduction in reductions:
            parse(image, cut, reduction, f"{output_folder_name}/layer_{reduction}x")