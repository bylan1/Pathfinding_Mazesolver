import math
import numpy as np
from skimage import io
import cv2

def load(img_path):
    """Loads an image from the file path using skimage.io.imread
    Converting pixel values between 0.0 and 1.0

    Inputs:
        img_path: file path to image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None
    out = io.imread(img_path)/255
    
    return out

def display(img, caption=''):
    """Displays an image given details

    Inputs:
        img: RGB image
        caption: image caption (blank otherwise)

    Returns:
        null
        Prints out image
    """
    plt.figure()
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')
    plt.show()

def print_stats(image):
    """Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: 
        none
        Prints stats of image shape
    """
    print("Image height: ", image.shape[0])
    print("Image width: ", image.shape[1])
    print("Number of channels: ", image.shape[2])
    
    return None

def find_points(image):
    """Scan image for coloured points

    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).

    Returns:
        coords: array of x and y location of coloured points in array
    """

    return None

def binary(input_image, threshold):
    """Converts a RGB image into binary

    Inputs:
        input_image: RGB image of shape(image_height, image_width, 3)
        threshold: limit value splitting binary 0 and 1 values

    Returns:
        out: binary numpy array with shape `(output_rows, output_cols)`
    """
    out = None

    temp = input_image.copy()

    for row in range(input_image.shape[0]):
        for col in range(input_image.shape[1]):
            temp[row, col] = np.multiply(temp[row, col] > threshold, 1)

    out = temp

    return out

