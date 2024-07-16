import os
import math
import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

### This Python program will store all of the functions used to handle the input and the necessary preprocessing of the image before beginning the pathfinding aspect of the project

# For image loading (0.0 to 1.0)
def load(img_path, extensions=['.png', '.jpg', '.jpeg']):
    """Loads an image from the file path using skimage.io.imread
    Try all file extensions
    Converting pixel values between 0.0 and 1.0

    Inputs:
        img_path: file path to image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    for ext in extensions:
        file_path = f'{img_path}{ext}'
        if os.path.isfile(file_path):
            out = io.imread(file_path)/255
            if out is not None:
                return out
            else:
                print(f"Failed to load image: {file_path}")
    raise FileNotFoundError(f"No valid image found for {img_path} with extensions {extensions}")
    
    return out

# For visual preparation
def image_prep(img, caption=''):
    plt.figure()
    plt.imshow(img, cmap='grey')
    plt.title(caption)
    plt.axis('off')

    ax = plt.gca()
    rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], linewidth=5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

# For visual analysis
def display(img, caption=''):
    """Displays an image given details

    Inputs:
        img: RGB image
        caption: image caption (blank otherwise)

    Returns:
        null
        Prints out image with border
    """
    image_prep(img, caption)

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
    print("Number of channels: ", image.shape[2] if len(image.shape) > 2 else 1)
    
    return None


# For image quality compression
def compress(image):
    """Compresses image into lower quality for convenient processing

    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels)

    Returns:
        out: numpy array of shape(image_width/2, image_height/2, n_channels)
    """

    out = cv2.resize(image, dsize = (int(image.shape[1]/7.5), int(image.shape[0]/7.5)), interpolation = cv2.INTER_NEAREST)

    return out

# For key point finding
def colour_process(image):
    """Save red and green point coordinates, remove others

    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels)

    Returns:
        coords: array of x and y location of coloured points in array
        out: image removing the red and green elements except key points
    """
    coords = []
    
    red_mask = (image[:, :, 0] > 0.5) & (image[:, :, 1] < 0.5) & (image[:, :, 2] < 0.75)
    green_mask = (image[:, :, 0] < 0.75) & (image[:, :, 1] > 0.5) & (image[:, :, 2] < 0.5)
    blue_mask = (image[:, :, 0] < 0.5) & (image[:, :, 1] < 0.75) & (image[:, :, 2] > 0.5)

    red_coords = np.argwhere(red_mask)
    green_coords = np.argwhere(green_mask)

    out = image.copy()

    if red_coords.size > 0:
        red_intensities = image[:, :, 0][red_mask]
        max_red_index = np.argmax(red_intensities)
        most_red_pixel = red_coords[max_red_index]
        coords.append(tuple(most_red_pixel))

    if green_coords.size > 0:
        green_intensities = image[:, :, 1][green_mask]
        max_green_index = np.argmax(green_intensities)
        most_green_pixel = green_coords[max_green_index]
        coords.append(tuple(most_green_pixel))

    out[red_mask | green_mask] = [1.0, 1.0, 1.0]
    out[blue_mask] = [1.0, 1.0, 1.0]

    if red_coords.size > 0:
        out[red_coords[0][0], red_coords[0][1]] = [1.0, 0, 0]
    if green_coords.size > 0:
        out[green_coords[0][0], green_coords[0][1]] = [0, 1.0, 0]
    
    return out, coords

# For binary conversion
def binary(image, threshold):
    """Converts a RGB image into binary

    Inputs:
        image: RGB image of shape(image_height, image_width, 3)
        threshold: limit value splitting binary 0 and 1 values

    Returns:
        out: binary numpy array with shape `(output_rows, output_cols)`
    """
    mean_vals = np.mean(image, axis=-1)

    out = np.zeros_like(mean_vals)
    out[mean_vals > threshold] = 1
    out[mean_vals <= threshold] = 1000000

    return out

# For all preprocessing methods
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

    temp = compress(image)
    temp, coords = colour_process(temp)
    out = binary(temp, threshold)

    return out, coords

# image1 = load('paths/path-1.png')
# display(image1, 'Blank Path with red and green points')
# coords = []

# # module test compress func
# # c_image = compress(image1)
# # display(c_image, 'Compressed image')

# new_image, coords = preprocess(image1, 0.5)
# print(coords)
# display(new_image, 'binary image with single red and green pixels')
# print_stats(new_image)