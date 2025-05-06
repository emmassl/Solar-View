import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern

'''Actionables: 
    1. make more thorough
    2. Implement with image display and information right below image
    3. Currently runs off of cropped image of roof, need to detect roof, presence of panels, then material
    4. Therefore fix issue of detection reliance on image quality and crop'''


''' Detecting Material Based on Local Binary Pattern (LBP) '''

def id_material_lbp(image_path):
    '''detects roof material using local binary pattern'''

    img = cv2.imread(image_path)
    # First convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Settings for LBP
    radius = 3 # radius of circular comparison
    n_points = 8 * radius # comparison with 8 neighbors
    METHOD = "uniform"
    
    # 1. Returns local_binary_pattern from skimage
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)

    # 2: Build histogram of LBP codes
        # np.histogram(a, bins, range, density, weights)
        # .ravel() returns flattens 1D array 
    (hist, _) = np.histogram(lbp.ravel(), np.arange(0, n_points + 3), (0, n_points + 2))

    # normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) # prevents division by 0

    # 3: Analyzing histogram shape
    texture_variance = np.var(hist)
    dominant_bin = np.argmax(hist)

    # 4: Based upon assumed rules of texture variance for roof materials, sort roofs
        # low small fraction varience: uniform or smoother texture, i.e. metal or concrete roof
    if texture_variance < 0.001:
        material = "metal or concrete (smooth)"
    elif dominant_bin == (n_points + 1): # most common pattern is non uniform? last bin. 
        if texture_variance > 0.005:
            material = "tile (patterned)"
        else:
            material = "asphalt or wood shingles (fine grain)"
    else:
        material = "unknown"
    
    return material

''' Detecting Material Based on SIFT '''
def id_material_sift(image_path):
    pass

if __name__ == "__main__":
    image_folder = "materials"
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(image_folder, filename)
            materiallbp = id_material_lbp(path)
            # print(f"{filename}: Detected roof material with SIFT: {materialsift}")
            print(f"{filename}: Detected roof material with LBP: {materiallbp}")

'''References

    LBP: 
    https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_local_binary_pattern.html
    https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html 
    https://stackoverflow.com/questions/39011167/why-does-the-local-binary-pattern-function-in-scikit-image-provide-same-value-fo 
    Biggest help: https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    numpy:
        https://numpy.org/doc/stable/reference/generated/numpy.histogram.html#numpy.histogram
        https://numpy.org/doc/2.1/reference/generated/numpy.var.html
        https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html

    SIFT: 

    '''