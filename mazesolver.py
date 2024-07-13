import sys
from inputhandler import *

# To run code, type in command line: python mazesolver.py [path_name] [method (optional)]
# path_name must be a image file stored in paths folder within mazesolver (mazesolver/paths/)
# For simple testing sake, run: python mazesolver.py path-1

# Choose a particular path, i.e. path-1 and input

maze_file = sys.argv[1]
test_image = load(f'paths/{maze_file}.png')

preproc_img, coords = preprocess(test_image, 0.5)

display(preproc_img, 'Preprocessed path')

