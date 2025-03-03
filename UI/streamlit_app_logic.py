from pic2music import PParser
import numpy as np
import streamlit as st
import cv2

def parse_music_sheet(image, progress_bar, params=None):
    parser = PParser()
    total_progress = 0
    
    progress_bar.progress(0)
    
    if params is None:
        params = {
            'resize_max_dim': 1200,
            'staff_dilate_iterations': 3,
            'staff_min_contour_area': 10000,
            'staff_pad_size': 0,
            'note_dilate_iterations': 2,
            'note_min_contour_area': 50,
            'note_pad_size': 0,
            'max_horizontal_distance': 10,
            'overlap_threshold': 0.8
        }
    
    image = parser.resize(image, max_dim=params['resize_max_dim'])
    transformed_image = parser.invert_colors(image)
    
    # Detect staff lines
    staff_line_contours = parser.find_contours(transformed_image, 
                                       dilate_iterations=params['staff_dilate_iterations'], 
                                       min_contour_area=params['staff_min_contour_area'], 
                                       pad_size=params['staff_pad_size'])
    
    staff_lines = parser.extract_contours(image, staff_line_contours, axis=1, full_height=False)
    total_steps = len(staff_lines) + 1 
    progress_increment = 100 / total_steps
    
    total_progress += progress_increment
    progress_bar.progress(int(total_progress))
    
    # Visualize the staff lines
    staff_lines_visualization = parser.draw_contours(image.copy(), staff_line_contours)
    cleaned_image = parser.remove_staff_lines(transformed_image)
    
    # Process each staff line
    all_notes = []
    all_notes_visualization = []
    for i, line in enumerate(staff_lines):
        # Get the cleaned version of this line
        cleaned_line = parser.extract_contours(cleaned_image, staff_line_contours, axis=1, full_height=False)[i]
        
        # Find note contours in this line
        note_contours = parser.find_contours(cleaned_line, 
                                        dilate_iterations=params['note_dilate_iterations'], 
                                        min_contour_area=params['note_min_contour_area'], 
                                        pad_size=params['note_pad_size'])
        note_contours = parser.group_note_components(note_contours, max_horizontal_distance=params['max_horizontal_distance'], overlap_threshold=params['overlap_threshold'])
        
        # Extract individual notes
        notes = parser.extract_contours(line, note_contours, axis=0, full_height=True)
        all_notes.append(notes)
        
        # Visualize notes on this line
        notes_visualization = parser.draw_contours(line.copy(), note_contours)
        all_notes_visualization.append(notes_visualization)
        # Update progress
        total_progress += progress_increment
        progress_bar.progress(int(total_progress))
    
    progress_bar.progress(100)
    
    return staff_lines_visualization, all_notes_visualization, all_notes