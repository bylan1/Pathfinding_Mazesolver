import math
import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt

# For image loading
def load(img_path):
    """Loads an image from the file path using skimage.io.imread
    Converting pixel values between 0.0 and 1.0

    Inputs:
        img_path: file path to image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None
    out = io.imread(img_path)
    
    return out

# For visual analysis
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

# For compression analysis
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

# For easier point finding
def high_saturation(image):
    out = image.copy()

    red_mask = (image[:, :, 0] > 127) & (image[:, :, 1] < 127) & (image[:, :, 2] < 127)
    green_mask = (image[:, :, 0] < 127) & (image[:, :, 1] > 127) & (image[:, :, 2] < 127)
    blue_mask = (image[:, :, 0] < 127) & (image[:, :, 1] < 127) & (image[:, :, 2] > 127)
    black_mask = (image[:, :, 0] < 127) & (image[:, :, 1] < 127) & (image[:, :, 2] < 127)
    white_mask = (image[:, :, 0] > 127) & (image[:, :, 1] > 127) & (image[:, :, 2] > 127)

    # Apply the masks to set the appropriate colors
    out[red_mask] = [255, 0, 0]
    out[green_mask] = [0, 255, 0]
    out[blue_mask] = [0, 0, 255]
    out[black_mask] = [0, 0, 0]
    out[white_mask] = [255, 255, 255]

    return out

# For key point finding
def find_points(image):
    """Scan image for red and green points

    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels)

    Returns:
        coords: array of x and y location of coloured points in array
    """
    coords = []
    red_c = 0
    green_c = 0

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            pixel = image[row, col]

            if pixel[0] > 127 and pixel[1] < 50 and pixel[2] < 50:
                image[row,col] = [255, 255, 255]
                if red_c == 0:
                    coords.append((row, col))
                    red_c = 1
            if pixel[0] < 50 and pixel[1] > 127 and pixel[2] < 50:
                image[row,col] = [255, 255, 255]
                if green_c == 0:
                    coords.append((row, col))
                    green_c = 1
    
    return coords

# def compress(image):
#     """Compresses image into lower quality for convenient processing

#     Inputs:
#         image: numpy array of shape(image_height, image_width, n_channels)

#     Returns:
#         out: numpy array of shape(image_height, image_width, n_channels)
#         identify appropriate height and width
#     """

def binary(input_image, threshold):
    """Converts a RGB image into binary

    Inputs:
        input_image: RGB image of shape(image_height, image_width, 3)
        threshold: limit value splitting binary 0 and 1 values

    Returns:
        out: binary numpy array with shape `(output_rows, output_cols)`
    """
    out = input_image.copy()

    for row in range(input_image.shape[0]):
        for col in range(input_image.shape[1]):
            out[row, col] = np.multiply(out[row, col] > threshold, 1)

    return out

def preprocess(image, threshold):
    """Applies preliminary point finding, compression and binary conversion to image

    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        threshold: limit value for binary preprocessing

    Returns:
        out: binary numpy array with shape `(output_rows, output_cols)`
        coords: array of x and y location of coloured points in array
    """
    out = None
    coords = []

    return out, coords

image1 = load('paths/blankpath.png')
image1 = high_saturation(image1)
display(image1, 'Blank Path with red and green points')
coords = find_points(image1)
print(coords)
display(image1, 'Blank Path with red and green points')