import cv2
import numpy as np
import imutils
from imutils import contours, perspective
from PIL import Image
import os
from typing import List, Tuple, Optional, Union, Any

class PParser:
    
    def imread(self, path: str) -> np.ndarray:
        """
        Load an image from a file path.
        
        Args:
            path (str): Path to the image file.
            
        Returns:
            numpy.ndarray: The loaded image in BGR format.
        """
        return cv2.imread(path)
    
    def imwrite(self, path: str, image: np.ndarray, overwrite: bool = False) -> bool:
        """
        Save an image to a file path.
        
        Args:
            path (str): Path where the image will be saved.
            image (numpy.ndarray): The image to save.
            overwrite (bool, optional): Whether to overwrite existing files without prompting. 
                                       Defaults to False.
        
        Returns:
            bool: True if the image was saved successfully, False otherwise.
        """
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
    
    def imshow(self, image: np.ndarray) -> None:
        """
        Display an image in a window until a key is pressed or the window is closed.
        
        Args:
            image (numpy.ndarray): The image to display.
        """
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)
        if key == 27 or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            
    def resize(self, image: np.ndarray, max_dim: int = 1200) -> np.ndarray:
        """
        Resize an image while maintaining aspect ratio.
        
        Args:
            image (numpy.ndarray): The image to resize.
            max_dim (int, optional): Maximum dimension (width or height) in pixels. Defaults to 1200.
            
        Returns:
            numpy.ndarray: The resized image.
            
        Note:
            Stores original and resized shapes as instance attributes.
        """  
        self.original_shape = image.shape
        if image.shape[0] <= max_dim and image.shape[1] <= max_dim:
            return image
        
        height, width = image.shape[:2]
        if height > width:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        else:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
            
        self.resized_shape = (new_width, new_height)
        
        return cv2.resize(image, (new_width, new_height))

    def invert_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Invert the colors of an image. Converts to grayscale if the image is in color.
        
        Args:
            image (numpy.ndarray): The image to invert.
            
        Returns:
            numpy.ndarray: The inverted image.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.bitwise_not(image)
    
    def find_contours(self, image: np.ndarray, dilate_iterations: int = 3, 
                      min_contour_area: int = 0, pad_size: int = 0) -> List[np.ndarray]:
        """
        Find contours in an image.
        
        Args:
            image (numpy.ndarray): The input image.
            dilate_iterations (int, optional): Number of dilation iterations to perform. Defaults to 3.
            min_contour_area (int, optional): Minimum area for a contour to be included. Defaults to 0.
            pad_size (int, optional): Padding to add around the image before processing. Defaults to 0.
            
        Returns:
            list: List of contours sorted from left to right.
        """
        padded_image = self._add_padding(image, pad_size)
        if len(image.shape) == 3:
            gray_line = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_line = padded_image.copy()
        _, binary_line = cv2.threshold(gray_line, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_image = cv2.dilate(binary_line, kernel, iterations=dilate_iterations)
        cnts = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = [c for c in cnts if cv2.contourArea(c) > min_contour_area]
        
        if pad_size:
            for cnt in cnts:
                cnt[:, :, 0] -= pad_size
                cnt[:, :, 1] -= pad_size 
        (cnts, _) = contours.sort_contours(cnts)
        
        return cnts
    
    def group_note_components(self, contours: List[np.ndarray], 
                             max_horizontal_distance: int = 10,
                             overlap_threshold: float = 0.8) -> List[np.ndarray]:
        """
        Group contours that likely belong to the same musical note based on horizontal proximity
        and containment.
        
        Args:
            contours (list): List of contours to group.
            max_horizontal_distance (int, optional): Maximum horizontal distance between 
                                                    contours to be considered part of the same group.
                                                    Defaults to 10.
            overlap_threshold (float, optional): Minimum overlap ratio to consider a contour as
                                               contained within another.
            
        Returns:
            list: List of merged contours where each group is represented as a single contour.
        """
        if not contours:
            return []
        
        def calculate_overlap(rect1, rect2):
            """Calculate the overlap ratio between two rectangles."""
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2
            
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
                
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            smaller_area = min(w1 * h1, w2 * h2)
            
            return intersection_area / smaller_area if smaller_area > 0 else 0.0
        
        groups = []
        used_contours = set()
        
        for i, cnt1 in enumerate(contours):
            if i in used_contours:
                continue
                
            current_group = [cnt1]
            rect1 = cv2.boundingRect(cnt1)
            
            for j, cnt2 in enumerate(contours[i+1:], start=i+1):
                if j in used_contours:
                    continue
                    
                rect2 = cv2.boundingRect(cnt2)
                overlap_ratio = calculate_overlap(rect1, rect2)
                
                x1, y1, w1, h1 = rect1
                x2, y2, w2, h2 = rect2
                horizontal_close = abs((x1 + w1) - x2) < max_horizontal_distance
                
                if horizontal_close or overlap_ratio > overlap_threshold:
                    current_group.append(cnt2)
                    used_contours.add(j)
            
            used_contours.add(i)
            groups.append(current_group)
        
        merged_contours = []
        for group in groups:
            if len(group) == 1:
                merged_contours.append(group[0])
            else:
                x_min = min(cv2.boundingRect(c)[0] for c in group)
                y_min = min(cv2.boundingRect(c)[1] for c in group)
                x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in group)
                y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in group)
                
                rect = np.array([[
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ]], dtype=np.int32)
                merged_contours.append(rect)
        
        return sorted(merged_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    def _add_padding(self, image: np.ndarray, pad_size: int = 0) -> np.ndarray:
        """
        Add padding around an image.
        
        Args:
            image (numpy.ndarray): The input image.
            pad_size (int, optional): Size of padding to add on all sides. Defaults to 0.
            
        Returns:
            numpy.ndarray: The padded image.
        """
        if pad_size > 0:
            padded = cv2.copyMakeBorder(
                image,
                pad_size, pad_size, pad_size, pad_size,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
            return padded
        return image
    
    def __mid_point(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

    def draw_contours(self, original_image: np.ndarray, cnts: List[np.ndarray], 
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2, show_midpoints: bool = False) -> np.ndarray:
        """
        Draw contours on an image with bounding boxes, midpoints, and cross lines.
        
        Args:
            original_image (numpy.ndarray): The image to draw on.
            cnts (list): List of contours to draw.
            color (tuple, optional): BGR color for the contour lines. Defaults to (0, 255, 0).
            thickness (int, optional): Thickness of the contour lines. Defaults to 2.
            
        Returns:
            numpy.ndarray: Image with drawn contours and annotations.
        """
        orig = original_image.copy()
        for c in cnts:
            # Get the minimum area rectangle
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            
            # Draw all elements in one pass
            cv2.drawContours(orig, [box.astype("int")], -1, color, thickness)
            
            if show_midpoints:
                # Calculate all midpoints 
                tl, tr, br, bl = box
                midpoints = [
                    self.__mid_point(tl, tr),  # top
                    self.__mid_point(bl, br),  # bottom
                    self.__mid_point(tl, bl),  # left
                    self.__mid_point(tr, br)   # right
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
    
    def extract_contours(self, image: np.ndarray, contours: List[np.ndarray], 
                        axis: int, full_height: bool = False) -> List[np.ndarray]:
        """
        Extract regions from an image based on contours, sorted by position.
        
        Args:
            image (numpy.ndarray): The source image.
            contours (list): List of contours to extract.
            axis (int): Sorting axis - 0 for horizontal (left to right), 1 for vertical (top to bottom).
            full_height (bool, optional): Whether to extract the full height of the image for each contour.
                                         Defaults to False.
            
        Returns:
            list: List of image regions corresponding to each contour.
            
        Raises:
            ValueError: If axis is not 0 or 1.
        """
        if axis == 0:
            sorted_cnts = sorted(contours, key=lambda c: cv2.boundingRect(c)[0]) 
        elif axis == 1:
            sorted_cnts = sorted(contours, key=lambda c: cv2.boundingRect(c)[1]) 
        else:
            raise ValueError("Axis must be 0 for horizontal or 1 for vertical")
        
        results = []
        img_height = image.shape[0]
        
        for c in sorted_cnts:
            x, y, w, h = cv2.boundingRect(c)
            if full_height:
                region = image[0:img_height, x:x+w]
            else:
                region = image[y:y+h, x:x+w] 
            results.append(region)
        return results
    
    def remove_staff_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Remove horizontal staff lines from a music score image.
        
        Args:
            image (numpy.ndarray): The input image, typically a music score.
            
        Returns:
            numpy.ndarray: Image with staff lines removed.
        """
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
        thresh = cv2.subtract(image, detected_lines)
        return thresh