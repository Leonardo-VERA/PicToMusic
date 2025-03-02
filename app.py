import streamlit as st
import time
import numpy as np 
from UI.streamlit_app_logic import parse_music_sheet
import cv2

st.set_page_config(
    page_title="Pic to Music App",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS 
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-message {
        padding: 1rem;
        background-color: #dff0d8;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 Pic to Music App")
st.markdown("""
    <div class='info-box'>
        Transform your music sheets into playable music! Simply upload an image or take a photo and we'll do the rest. 🎵
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Upload File", "📸 Take Photo"])

# Tab 1: Upload File
with tab1:
    st.markdown("### 📤 Upload Your Music Sheet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            try:
                st.markdown("""
                    <div class='success-message'>
                        ✅ File uploaded successfully! Click 'Parse Music Sheet' when ready.
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with col2:
        if uploaded_file is not None:
            with st.spinner("Processing image..."):
                st.image(uploaded_file, caption="Preview")

# Tab 2: Take Photo
with tab2:
    st.markdown("### 📸 Capture Music Sheet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        camera_input = st.camera_input(
            "Take a picture",
            help="Make sure the sheet music is well-lit and clearly visible"
        )
        
        if camera_input is not None:
            try:
                st.markdown("""
                    <div class='success-message'>
                        ✅ Image captured successfully! Click 'Parse Music Sheet' when ready.
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        if camera_input is not None:
            with st.spinner("Processing image..."):
                st.image(camera_input, caption="Preview")

if camera_input is not None or uploaded_file is not None:
    st.markdown("---")
    if st.button("🎵 Parse Music Sheet"):
        st.subheader("Parsing results...")
        with st.spinner("🎼 Converting your sheet music..."):
            try:
                # Get the image from either source
                image_source = uploaded_file if uploaded_file is not None else camera_input
                
                # Convert to numpy array
                image_bytes = image_source.getvalue()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Initialize progress bar
                progress_bar = st.progress(0)
                
                # Process the image
                staff_visualization, notes = parse_music_sheet(image, progress_bar)
                
                # Display results in columns
                st.success("✨ Music sheet successfully parsed!")

                st.markdown("""
                    <div class='info-box'>
                        Processing complete! Available options:
                        - ▶️ Play the converted music
                        - 💾 Download as MIDI file
                        - 🎼 View the musical notation
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing sheet music: {str(e)}")
                st.markdown("""
                    <div class='info-box' style='background-color: #ffebee;'>
                        Tips for better results:
                        - Ensure the image is clear and well-lit
                        - Make sure the sheet music is properly aligned
                        - Try adjusting the image contrast
                    </div>
                """, unsafe_allow_html=True)
        if st.button("🎵 Convert to Music"):
            st.subheader("Converting to music...")
            with st.spinner("🎼 Converting to music..."):
                pass
        

# Add helpful information at the bottom
with st.expander("ℹ️ Tips for Best Results"):
    st.markdown("""
        ### 📝 Guidelines for Best Results
        
        #### When Uploading Files:
        - Use high-resolution images
        - Ensure the sheet music is well-lit
        - Avoid glare or shadows on the page
        - Make sure the entire sheet is visible
        
        #### When Taking Photos:
        - Hold your device steady
        - Use good lighting
        - Avoid shadows
        - Center the sheet music in frame
        - Keep the camera parallel to the page
        
        ### 🎵 Supported Features
        - Standard musical notation
        - Multiple staves
        - Various time signatures
        - Different key signatures
    """)