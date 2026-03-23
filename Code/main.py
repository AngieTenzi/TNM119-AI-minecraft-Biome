import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import data, exposure, io, color, transform


def extract_features(image):
    #Resize image
    image = transform.resize(image, (320, 180))

    # HOG features
    gray = color.rgb2gray(image)
    hog_features = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        feature_vector=True,
        visualise=True,
    )

    # Color histogram
    hsv_image = color.rgb2hsv(image)
    hist_hue, _ = np.histogram(hsv_image[:,:,0], bins=8, range=(0, 255))
    hist_sat, _ = np.histogram(hsv_image[:, :, 1], bins=8, range=(0, 255))
    hist_val, _ = np.histogram(hsv_image[:, :, 2], bins=8, range=(0, 255))


image = "../BiomeData/biome1/biome_1_0.jpg"
