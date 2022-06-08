import cv2 as cv
from autres_fonctions import load

def load(filename):
    return cv.imread(filename)

# 
# Fonction resize
# 
def resize_image(img,new_dim):
    return cv.resize(img,new_dim)



# 
# Fonction rotate
# 
def rotate_image(img,degree):
    rows,cols,depth=img.shape
    center=(cols/2,rows/2)
    M=cv.getRotationMatrix2D(center,degree,1)
    img_rotate=cv.warpAffine(img,M,(cols,rows))
    return img_rotate



# 
# Fonction smoothing
# 
def smoothing_image(img,ksize):
    return cv.GaussianBlur(img,ksize,0) 



# 
# draw_rectangles
#
def draw_rectangle(img,tlcorner,brcorner,color,line_thickness):
    return cv.rectangle(img,tlcorner,brcorner,color,line_thickness)


def grey_image(img):
    new_image=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return new_image
