"""
This is the main file with all functions for the jigsaw puzzle solver. 

The original author of this file is Maxim Terleev. The file was retrieved from the following GitHub repository:
https://github.com/MaximTerleev/Jigsaw-Puzzle-AI/blob/main/jigsaw-puzzle-solver.py

The solver contains many functions and can be split into 3 main sections:
1. Image Processing
2. Matching
3. Assembling

"""

# imports

from scipy.ndimage import filters
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import numpy as np
import cv2
