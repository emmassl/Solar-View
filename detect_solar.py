import os
import cv2
import numpy as np



def detect_hsv_blue(img):
    """Detects solar panels using HSV color segmentation (blue hue range)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define bounds of blue panels
    lower_blue = np.array([90, 40, 20])
    upper_blue = np.array([135, 255, 130])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Double Check.
    panel_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Remove small and big shapes
        if area < 400 or area > 30000:
            continue
        # Check rectangles again
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) < 4 or len(approx) > 6:
            continue
        # Check ratio of size
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.6 or aspect_ratio > 2.1:
            continue
        
        panel_contours.append(cnt)

    return panel_contours

def detect_solar_panels(image_path, show=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return "unreadable"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 160)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Evaluating potential panel existence
    panel_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 1. Remove very small shapes
        if area < 450:
            continue
        # 2. Approximate sides of panels (rectangle)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) < 4 or len(approx) > 6:
            continue
        # 3. Approximate general shape ratio of panels
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.6 or aspect_ratio > 2.1: 
            continue
        # 4. Check darkness inside region (usually panels)
        roi = gray[y:y+h, x:x+w]
        mean_intensity = np.mean(roi)
        if mean_intensity > 120:  # not dark enough
            continue

        panel_candidates.append(cnt)

    # Double-check: Use HSV-based color detection if no panels found
    if not panel_candidates:
        panel_candidates = detect_hsv_blue(img)

    # if panel_candidates and show:
    # #     cv2.drawContours & cv2.imshow
    return "panels detected" if panel_candidates else "no panels detected"



if __name__ == "__main__":
    image_folder = "buildings"
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(image_folder, filename)
            result = detect_solar_panels(path)
            print(f"{filename}: {result}")



"""References: 

        https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
        https://www.geeksforgeeks.org/python-opencv-canny-function/
        https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
    HSV: 
        https://medium.com/neurosapiens/segmentation-and-classification-with-hsv-8f2406c62b39
        really helpful: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html 

"""
