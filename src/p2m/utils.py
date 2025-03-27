import numpy as np
import pandas as pd
import cv2
import csv
from pathlib import Path
from typing import List, Tuple
from p2m.scoretyping import StaffLine

def imreshape(image: np.ndarray, shape: int = 128):
    return cv2.resize(image, (shape, shape))

def generate_detection_csv(staff_lines_list: List[List[StaffLine]], 
                         output_filename: str, include_staff: bool = False) -> None:
    """
    Generate a detection CSV file from multiple lists of staff lines and their notes.
    
    Args:
        staff_lines (List[StaffLine]): List of staff line lists, one for each score
        output_filename (str): Path to output CSV file
    """
     
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        
        for staff_lines in staff_lines_list:
            for staff in staff_lines:
                if include_staff:
                    x, y, w, h = staff.bounds
                    width = staff.image.shape[1]
                    height = staff.image.shape[0]
                    label = "staff"
                    
                    writer.writerow([
                        staff.filename, # filename
                        width,          # width
                        height,         # height
                        label,          # class
                        x,              # xmin
                        y,              # ymin
                        x + w,          # xmax
                        y + h           # ymax
                    ])                
                for note in staff.notes:
                    x, y, w, h = note.full_height_bounds
                    width = note.image.shape[1]
                    height = note.image.shape[0]
                    label = note.label if note.label else "note"
                    
                    writer.writerow([
                        staff.filename, # filename
                        width,          # width
                        height,         # height
                        label,          # class
                        x,              # xmin
                        y,              # ymin
                        x + w,          # xmax
                        y + h           # ymax
                    ])