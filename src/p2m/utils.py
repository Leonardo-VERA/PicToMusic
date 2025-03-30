import numpy as np
import pandas as pd
import cv2
import csv
from pathlib import Path
from typing import List, Tuple
from p2m.scoretyping import StaffLine
import zipfile
import os
from tqdm import tqdm
import loguru
import shutil

def imreshape(image: np.ndarray, shape: int = 128):
    return cv2.resize(image, (shape, shape))

def generate_detection_csv(staff_lines_list: List[List[StaffLine]], 
                         output_filename: str, include_staff: bool = False) -> None:
    """
    Generate a detection CSV file from multiple lists of staff lines and their notes.
    
    Args:
        staff_lines (List[StaffLine]): List of staff line lists, one for each score
        output_filename (str): Path to output CSV file
        include_staff (bool): Whether to include staff lines in the CSV
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
                    

def extract_dataset(zip_path: str):
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Could not find {zip_path}")
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        loguru.logger.info(f"Extracting {zip_path}")
        total_size = sum(file.file_size for file in zip_ref.filelist)
        loguru.logger.info(f"Total size: {total_size} bytes")
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
            for file in zip_ref.filelist:
                zip_ref.extract(file)
                pbar.update(file.file_size)
    
    loguru.logger.info(f"Successfully extracted {zip_path}")
    
def split_data(input_path: str, output_path: str, batch_size: int, num_batch: int):
    """
    Split the dataset into multiple batches and save them to the output path.
    
    Args:
        input_path (str): Path to the input dataset
        output_path (str): Path to save the batches
        batch_size (int): Number of files per batch
        num_batch (int): Number of batches to create
    """

    os.makedirs(output_path, exist_ok=True)
    
    files = os.listdir(input_path)
    
    for batch_num in tqdm(range(num_batch), desc="Processing batches", total=num_batch, unit="batch"):
        batch_folder = os.path.join(output_path, f'batch_{batch_num}')
        os.makedirs(batch_folder, exist_ok=True)
        
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(files))
        
        for file_name in files[start_idx:end_idx]:
            src = os.path.join(input_path, file_name)
            dst = os.path.join(batch_folder, file_name)
            shutil.copy2(src, dst)
    loguru.logger.info(f"Successfully processed {num_batch} batches and saved to {output_path}")
    return output_path