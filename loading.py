# Load training images from directory.
# Create two lists X, and y. 
# X is a list of length num_training_images containing equally sized, normalized, greyscale training images.
# y is a list of length num_training_images containing integer values 0 or 1. 
# If X[i] is a dog y[i] = 1
# If X[i] is a cat y[i] = 0

import os
from telnetlib import X3PAD
import cv2

X = []
y = []

folder_dir = "./train"

# Formatting all images into X
for images in os.listdir(folder_dir):
    # if (images.endswith(".jpg")):
    #     print(images)
    
    image = cv2.imread("./train/"+images)
    print("/train/"+images)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    resized = cv2.resize(image, (100,100))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    final = gray/255
    X.append(final)
    
# detect dog or cat, set 0 or 1 in list y
for i in X:
    break;