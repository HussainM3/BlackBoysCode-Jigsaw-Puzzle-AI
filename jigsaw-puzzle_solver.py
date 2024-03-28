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

"""# Image processing"""

# @title Functions

def showpic(image, width=10):
  plt.figure(figsize=(width, width/1000*727))
  plt.imshow(image, cmap='gray')
  plt.axis('off')
  plt.show()

def showlist(tiles, width=10):
  n_rows = np.ceil(len(tiles)/5).astype('int')
  plt.subplots(n_rows, 5, figsize=(width, width))
  for i in range(len(tiles)):
    plt.subplot(n_rows, 5, i+1)
    plt.axis('off')
    plt.title(str(i))
    plt.imshow(tiles[i])
  plt.show()

# Load scanned tiles
puzzle = np.array(Image.open('puzzle.png').convert('RGBA'))
print(puzzle.shape)
showpic(puzzle)

# Adaptive thresholding
thresh = cv2.cvtColor(puzzle, cv2.COLOR_RGBA2GRAY)
thresh = cv2.adaptiveThreshold(thresh, 255, 0, 1, 3, 3)
thresh = cv2.GaussianBlur(thresh, (3,3), 1)
showpic(thresh)

# Find and fill contours
contours, _ = cv2.findContours(thresh, 0, 1)
sorting = sorted([[cnt.shape[0], i] for i, cnt in enumerate(contours)], reverse=True)[:15]
biggest = [contours[s[1]] for s in sorting] 
fill = cv2.drawContours(np.zeros(puzzle.shape[:2]), biggest, -1, 255, thickness=cv2.FILLED)
showpic(fill)

# Smooth contours and trim shadows
smooth = filters.median_filter(fill.astype('uint8'), size=10)
trim_contours, _ = cv2.findContours(smooth, 0, 1)
cv2.drawContours(smooth, trim_contours, -1, color=0, thickness=1)
showpic(smooth)

# Split into tiles
contours, _ = cv2.findContours(smooth, 0, 1)
tiles, tile_centers = [], []
for i in range(len(contours)):
  x, y, w, h = cv2.boundingRect(contours[i])
  shape, tile = np.zeros(puzzle.shape[:2]), np.zeros((300,300,4), 'uint8')
  cv2.drawContours(shape, [contours[i]], -1, color=1, thickness=-1)
  shape = (puzzle * shape[:,:,None])[y:y+h,x:x+w,:]
  tile[(300-h)//2:(300-h)//2+h,(300-w)//2:(300-w)//2+w] = shape
  tiles.append(tile)
  tile_centers.append((h//2+y, w//2+x))

showlist(tiles)

