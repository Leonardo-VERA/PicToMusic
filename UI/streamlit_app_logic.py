from pic2music import PParser
import numpy as np
import streamlit as st

def parse_music_sheet(image, progress_bar):
    parser = PParser()
    total_progress = 0
    
    progress_bar.progress(0)
    image = parser.resize(image)
    transformed_image = parser.invert_colors(image)
    
    # Detect staff lines
    staff_line_contours = parser.find_contours(transformed_image, 
                                        dilate_iterations=3, 
                                        min_contour_area=10000, 
                                        pad_size=0)
    
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
    for i, line in enumerate(staff_lines):
        # Get the cleaned version of this line
        cleaned_line = parser.extract_contours(cleaned_image, staff_line_contours, axis=1, full_height=False)[i]
        
        # Find note contours in this line
        note_contours = parser.find_contours(cleaned_line, 
                                        dilate_iterations=2, 
                                        min_contour_area=50, 
                                        pad_size=0)
        note_contours = parser.group_note_components(note_contours, max_horizontal_distance=5)
        
        # Extract individual notes
        notes = parser.extract_contours(line, note_contours, axis=0, full_height=True)
        all_notes.append(notes)
        
        # Visualize notes on this line
        notes_visualization = parser.draw_contours(line.copy(), note_contours)
        st.image(notes_visualization, caption=f"Staff Line {i+1} Analysis")
        
        # Update progress
        total_progress += progress_increment
        progress_bar.progress(int(total_progress))
    
    progress_bar.progress(100)
    
    return staff_lines_visualization, all_notes
