import os
import cv2
from detect_solar import detect_solar_panels 
from detect_material import id_material_lbp, id_material_sift

def center_crop(img, percent):
    '''crop central (25%) for material detection, assuming roof is centered.'''
    h, w = img.shape[:2]
    ch, cw = int(h * percent), int(w * percent)
    x_start = max((w - cw) // 2, 0)
    y_start = max((h - ch) // 2, 0)
    return img[y_start:y_start+ch, x_start:x_start+cw]


if __name__ == "__main__":
    dataset = "images"
    for filename in os.listdir(dataset):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(dataset, filename)
            print(f"\nProcessing: {filename}")

            result = detect_solar_panels(path)
            if result == "panels detected":
                print("Solar panels detected")
            else:
                print("No panels detected. Running material detection:")
                img = cv2.imread(path)
                if img is None:
                    print("Could not read image")
                    continue
                gray = cv2.cvtColor(center_crop(img, 0.25), cv2.COLOR_BGR2GRAY)

                # Run material detection (both LBP and SIFT) on cropped image
                material_lbp, _ = id_material_lbp(gray)
                sift = cv2.SIFT_create()
                material_sift, _ = id_material_sift(gray, sift)

                print(f"LBP Detected Roof Material: {material_lbp}")
                print(f"SIFT Detected Roof Material: {material_sift}")
