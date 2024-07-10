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
    out = io.imread(img_path)/255
    
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

    red_mask = (image[:, :, 0] > 0.5) & (image[:, :, 1] < 0.5) & (image[:, :, 2] < 0.5)
    green_mask = (image[:, :, 0] < 0.5) & (image[:, :, 1] > 0.5) & (image[:, :, 2] < 0.5)
    blue_mask = (image[:, :, 0] < 0.5) & (image[:, :, 1] < 0.5) & (image[:, :, 2] > 0.5)
    black_mask = (image[:, :, 0] < 0.5) & (image[:, :, 1] < 0.5) & (image[:, :, 2] < 0.5)
    white_mask = (image[:, :, 0] > 0.5) & (image[:, :, 1] > 0.5) & (image[:, :, 2] > 0.5)

    # Apply the masks to set the appropriate colors
    out[red_mask] = [1.0, 0, 0]
    out[green_mask] = [0, 1.0, 0]
    out[blue_mask] = [0, 0, 1.0]
    out[black_mask] = [0, 0, 0]
    out[white_mask] = [1.0, 1.0, 1.0]

    return out

# For key point finding
def find_points(image):
    """Scan image for red and green points

    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels)

    Returns:
        coords: array of x and y location of coloured points in array
        out: image removing the red and green elements except key points
    """
    coords = []
    
    red_mask = (image[:, :, 0] > 0.5) & (image[:, :, 1] < 0.5) & (image[:, :, 2] < 0.5)
    green_mask = (image[:, :, 0] < 0.5) & (image[:, :, 1] > 0.5) & (image[:, :, 2] < 0.5)

    red_coords = np.argwhere(red_mask)
    green_coords = np.argwhere(green_mask)

    out = image.copy()

    coords.append(tuple(red_coords[0]))
    out[red_coords[0][0], red_coords[0][1]] = [1.0, 0, 0]

    coords.append(tuple(green_coords[0]))
    out[green_coords[0][0], green_coords[0][1]] = [0, 1.0, 0]

    out[red_mask | green_mask] = [1.0, 1.0, 1.0]
    
    return out, coords

# def compress(image):
#     """Compresses image into lower quality for convenient processing

#     Inputs:
#         image: numpy array of shape(image_height, image_width, n_channels)

#     Returns:
#         out: numpy array of shape(image_height, image_width, n_channels)
#         identify appropriate height and width
#     """

def binary(image, threshold):
    """Converts a RGB image into binary

    Inputs:
        image: RGB image of shape(image_height, image_width, 3)
        threshold: limit value splitting binary 0 and 1 values

    Returns:
        out: binary numpy array with shape `(output_rows, output_cols)`
    """

    greyscale_image = np.mean(image, axis=-1)
    out = np.where(greyscale_image > threshold, 1, 0)
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
    temp = None
    coords = []

    temp = high_saturation(image)
    temp, coords = find_points(temp)
    out = binary(temp, threshold)

    return out, coords

image1 = load('paths/path-1.png')
display(image1, 'Blank Path with red and green points')
new_image, coords = preprocess(image1, 0.5)
print(coords)
display(new_image, 'binary image with single red and green pixels')