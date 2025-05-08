import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


def id_material_lbp(gray):
    ''' Detecting Material Based on Local Binary Pattern (LBP) '''
    # Settings for LBP
    radius = 3 # radius of circular comparison
    n_points = 8 * radius # comparison with 8 neighbors
    method = "uniform"

    # 1. Returns local_binary_pattern img from skimage
    lbp = local_binary_pattern(gray, n_points, radius, method)

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
    return material, lbp



def get_sift_features(gray, sift):
    # sift.detect(image, optional mask to limit detection to a specific area)
    keypoints = sift.detect(gray, None)
    # total number of pixels in the image, 2D Numpy Array, height*width
    area = gray.shape[0] * gray.shape[1]

    # calculating density of the keypoints! low keypoint count = smooth texture
    # 1e-5 again to avoid division by zero, though should never be. from img checks.
    kp_density = len(keypoints) / (area + 1e-5)
    # kp.pt gives coordinates.
    locations = np.array([kp.pt for kp in keypoints]) if keypoints else np.array([])
    # spread tells distribution of keypoints
    x_spread = np.std(locations[:, 0]) if len(locations) > 0 else 0
    y_spread = np.std(locations[:, 1]) if len(locations) > 0 else 0
    return keypoints, kp_density, x_spread, y_spread

# possibility to refine further for accuracy
def id_material_sift(gray, sift):
    ''' Detecting Material Based on SIFT '''
    keypoints, kp_density, x_spread, y_spread = get_sift_features(gray, sift)
    if kp_density > 0.002 and x_spread > 30 and y_spread > 30:
        material = "tile (regular pattern with high texture)"
    elif kp_density < 0.0005:
        material = "metal or concrete (smooth)"
    elif 0.0005 <= kp_density <= 0.002:
        material = "asphalt or wood shingles (medium, fine grain)"
    else:
        material = "unknown"
    outimg = cv2.drawKeypoints(gray, keypoints, None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return material, cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)



if __name__ == "__main__":
    image_folder = "materials"
    sift = cv2.SIFT_create()

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(image_folder, filename)
            # Get image
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to load image: {path}")
                continue

            img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # convert to gray for lbp and sift
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray)  # Improve contrast

            materiallbp, lbp_img = id_material_lbp(gray) #img is for display
            materialsift, sift_img = id_material_sift(gray, sift)

            # Display results with img and material detection
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            axes[0].imshow(img_color)
            axes[0].set_title("Input")
            axes[1].imshow(lbp_img)
            axes[1].set_title(f"LBP\n{materiallbp}")
            axes[2].imshow(sift_img)
            axes[2].set_title(f"SIFT\n{materialsift}")
            plt.show()



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
    https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
        matplot:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

    '''