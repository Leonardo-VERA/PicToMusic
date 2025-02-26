import cv2
import numpy as np
import imutils
from imutils import contours, perspective
from PIL import Image
import os

class PParser:
    
    def __init__(self, config=None):
        # Default hyperparameters
        self.config = {
            'gaussian_kernel': (5, 5),  # Blur kernel size
            'gaussian_sigma': 0,        # Blur sigma
            'morph_kernel': (3, 3),     # Morphological kernel size
            'min_contour_area': 100,    # Minimum contour area
            'dilate_iterations': 2,     # Dilation iterations
            'resize_max_dim': 1000,     # Maximum dimension for resize
        }
        
        if config:
            self.config.update(config)
    
    def imread(self, path):
        return cv2.imread(path)
    
    def imwrite(self, path, image, overwrite=True):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        if os.path.exists(path):
            if not overwrite:
                user_input = input(f"File {path} already exists. Overwrite? (y/n): ")
                if user_input.lower() != 'y':
                    print("Saving cancelled.")
                    return False
        
        return cv2.imwrite(path, image)
    
    def imshow(self, image):
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)
        if key == 27 or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
    
    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.config['gaussian_kernel'], 
                                  self.config['gaussian_sigma'])
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config['morph_kernel'])
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh
    
    def find_contours(self, image):
        dilated_image = cv2.dilate(image, None, 
                                  iterations=self.config['dilate_iterations'])
        cnts = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, 
                               cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = [c for c in cnts if cv2.contourArea(c) > self.config['min_contour_area']]
        cnts = self.__sort_contours(cnts)
        return cnts
    
    def __sort_contours(self, cnts):
        (cnts, _) = contours.sort_contours(cnts)
        return cnts
    
    def __sort_by_contour_dimensions(self, contours):
        return sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    def _mid_point(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)
    
    def draw_contours(self, image, contours, color=(0, 255, 0), thickness=2):
        orig = image.copy()
        for c in contours:
            # Get the minimum area rectangle
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            
            # Draw all elements in one pass
            cv2.drawContours(orig, [box.astype("int")], -1, color, thickness)
            
            # Calculate all midpoints 
            tl, tr, br, bl = box
            midpoints = [
                self._mid_point(tl, tr),  # top
                self._mid_point(bl, br),  # bottom
                self._mid_point(tl, bl),  # left
                self._mid_point(tr, br)   # right
            ]
            
            # Draw all circles and lines in batch
            for point in midpoints:
                cv2.circle(orig, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
            
            # Draw cross lines
            cv2.line(orig, (int(midpoints[0][0]), int(midpoints[0][1])),
                    (int(midpoints[1][0]), int(midpoints[1][1])), (255, 0, 255), 2)
            cv2.line(orig, (int(midpoints[2][0]), int(midpoints[2][1])),
                    (int(midpoints[3][0]), int(midpoints[3][1])), (255, 0, 255), 2)
        
        return orig
    
    def resize(self, image, max_dim=None):
        if max_dim is None:
            max_dim = self.config['resize_max_dim']
        
        if image.shape[0] <= max_dim and image.shape[1] <= max_dim:
            return image
        
        height, width = image.shape[:2]
        if height > width:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        else:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        
        return cv2.resize(image, (new_width, new_height))

if __name__ == "__main__":
    custom_config = {
        'gaussian_kernel': (7, 7),
        'gaussian_sigma': 1,
        'morph_kernel': (5, 5),
        'min_contour_area': 200,
        'dilate_iterations': 3,
        'resize_max_dim': 1200
    }
    
    parser = PParser(config=custom_config) 
    image = parser.imread("resources/samples/fire.jpg")
    image = parser.resize(image)  
    thresh = parser.preprocess(image)
    cnts = parser.find_contours(thresh)
    result = parser.draw_contours(image, cnts)
    parser.imwrite("resources/output/test/fire.jpg", result)
    # parser.imshow(result)
