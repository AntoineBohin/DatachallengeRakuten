import cv2
import argparse
import numpy as np
from autres_fonctions import *
from image_processing_operations import *



def apply_processing(image,process):
    imagedest=image
    if process == "resizing":
        new_size = input("What is the new size (two numbers separated by a space)  ? ")
        dimension = tuple([int(x) for x in new_size.split()])
        imagedest = resize_image(image,dimension)

    elif process == "rotating":
        rot_angle = int(input("What is the rotation angle (in degrees) ? "))
        imagedest = rotate_image(image, rot_angle)

    elif process == "smoothing":
        smooth_kernel = input("What is the filter size (two numbers separated by a space) ? ")
        kernel_size = tuple([int(x) for x in smooth_kernel.split()])
        imagedest = smoothing_image(image, kernel_size)

    elif process == "draw_rectangle":
        tl_corner = input("What is the position of the top left corner of the rectangle (two numbers separated by a space) ? ")
        br_corner = input("What is the position of the bottom right corner of the rectangle (two numbers separated by a space) ? ")
        color = input("What is the color of your rectangle (BGR, three calues separated by space) ? ")
        thickness = input("What is the thickness of the rectangle ? ")

        tl_position = tuple([int(x) for x in tl_corner.split()])
        br_position = tuple([int(x) for x in br_corner.split()])
        color = tuple([int(x) for x in color.split()])
        size = int(thickness)
        imagedest = draw_rectangle(image, tl_position, br_position, color, size)
    return imagedest
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",required=True, help="path to input image")
    ap.add_argument("-o", "--output", required=True, help="path to output image")
    ap.add_argument("-p", "--processing", required=True, help="processing mode ('resizing', 'rotating', 'smoothing', 'draw_rectangle')")
    args = vars(ap.parse_args())
    image_source_filename = args["input"]
    image_dest_filename = args["output"]
    process_type = args["processing"]
    image_source = load(image_source_filename)
    dest = apply_processing(image_source, process_type)
    cv2.imwrite(image_dest_filename, dest)
