import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import tempfile
import os
from UI.statics import apply_custom_css, create_file_uploader, create_camera_input
from sonatabene.parser import PParser
from ultralytics import YOLO
from sonatabene.converter import yolo_to_abc, abc_to_midi, abc_to_musescore, abc_to_audio, abc_to_musescore
from sonatabene.model import predict_with_api
from sonatabene.converter.converter_abc import INSTRUMENT_MAP
from sonatabene.utils import get_musescore_path
from midi2audio import FluidSynth
import pickle

# Session states
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'image' not in st.session_state:
    st.session_state.image = None
if 'image_color' not in st.session_state:
    st.session_state.image_color = None
if 'staves' not in st.session_state:
    st.session_state.staves = None
if 'staff_visualization' not in st.session_state:
    st.session_state.staff_visualization = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'abc_code' not in st.session_state:
    st.session_state.abc_code = None


st.set_page_config(
    page_title="Sonatabene - Demo",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_custom_css()

st.title("🎼 Sonatabene : From Paper to Music")
st.markdown("""
### Step-by-Step Guide
1. **Upload an image** of sheet music or take a photo
2. **Detect staff lines** and adjust parameters
3. **Classify notes** using our AI model
4. **Convert to ABC notation**
5. **Generate audio** and download various formats
""")

# Step 1: Image Loading
if st.session_state.step >= 1:
    st.title("Step 1: Input Image")
    tab1, tab2 = st.tabs(["📁 Upload File", "📸 Take Photo"])
    with tab1:
        uploaded_file = create_file_uploader()
    with tab2:
        camera_input = create_camera_input()

    if st.session_state.step == 1 and (camera_input is not None or uploaded_file is not None):
        try:
            if camera_input is not None:
                st.session_state.image = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            else:
                st.session_state.image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.success("✅ Image loaded successfully!")

            st.session_state.image_color = cv2.cvtColor(st.session_state.image, cv2.COLOR_RGB2BGR)
            
            st.session_state.step = 2
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

# Step 2: Note Classification
if st.session_state.step >= 2 and st.session_state.image is not None:
    st.title("Step 2: Note Classification")

    model = YOLO(model="models/chopin.pt")

    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
    with col2:
        nms_threshold = st.slider(
            "NMS Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Non-Maximum Suppression threshold"
        )

    if st.button("🎵 Classify Notes"):
        with st.spinner("Classifying notes..."):
            st.session_state.predictions = []

            result = model.predict(source=st.session_state.image_color, 
                            conf=confidence_threshold,
                            iou=nms_threshold,
                            save=False)[0]
            st.session_state.predictions.append(result)
            
            st.image(result.plot(), 
                    caption=f"Note Classification",
                    use_container_width=True)
                
            st.success("✅ Notes classified successfully!")
            st.session_state.step = 3

# Step 3: ABC Notation
if st.session_state.step >= 3 and st.session_state.predictions is not None:
    st.title("Step 3: ABC Notation")
    
    if st.button("🎼 Convert to ABC"):
        with st.spinner("Converting to ABC notation..."):
            st.session_state.abc_code = yolo_to_abc(st.session_state.predictions)
            st.code(st.session_state.abc_code, language="abc")
            st.success("✅ ABC notation generated!")
            st.session_state.step = 4

# Step 4: Audio Generation
if st.session_state.step >= 4 and st.session_state.abc_code is not None:
    st.title("Step 4: Audio Generation")
    
    col1, col2 = st.columns(2)
    with col1:
        instrument_name = st.selectbox("Choose an Instrument", list(INSTRUMENT_MAP.keys()),
                                     help="Select the instrument for audio generation")
    with col2:
        tempo = st.number_input("Select a Tempo", min_value=40, max_value=200, value=120, step=5,
                         help="Adjust the playback tempo")

    if st.button("🎵 Generate Audio", type="primary"):
        with st.spinner("Generating audio..."):
            try:
                instrument_class = INSTRUMENT_MAP[instrument_name]
                midi_buffer = BytesIO()
                abc_to_midi(st.session_state.abc_code, midi_buffer, instrument=instrument_class, tempo_bpm=tempo)
                midi_buffer.seek(0)
                midi_data = midi_buffer.read()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as midif:
                    midif.write(midi_data)
                    midi_path = midif.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wavf:
                    wav_path = wavf.name
                    soundfont_path = os.path.abspath(".fluidsynth/default_sound_font.sf2")
                    fs = FluidSynth(sound_font=soundfont_path)
                    fs.midi_to_audio(midi_path, wav_path)
                    with open(wav_path, 'rb') as wav_file:
                        results_audio = wav_file.read()
                
                st.audio(wav_path, format="audio/wav")
                st.success("✅ Audio generated successfully!")

                # MuseScore Option
                try:
                    musescore_path = get_musescore_path()
                    st.button("🎼 Open in MuseScore", 
                            on_click=lambda: abc_to_musescore(st.session_state.abc_code, open=True, musescore_path=musescore_path, instrument=instrument_class), 
                            use_container_width=True)
                except (FileNotFoundError, OSError) as e:
                    st.info(f"⚠️ {str(e)} [Click here to download MuseScore 4](https://musescore.org/en/download)")

                # Download Options
                st.subheader("Download Options")
                dl1, dl2, dl3 = st.columns(3)

                with dl1:
                    results_pickle = pickle.dumps(st.session_state.predictions)
                    st.download_button(
                        label="💾 Download YOLO Classification Data",
                        data=results_pickle,
                        file_name=f"{st.session_state['file_name']}_yolo_classification.pkl",
                        mime="application/octet-stream",
                        help="Download the YOLO classification results in pickle format",
                        use_container_width=True
                    )

                with dl2:
                    st.download_button(
                        label="🎵 Download MIDI File",
                        data=midi_data,
                        file_name=f"{st.session_state['file_name']}.mid",
                        mime="audio/midi",
                        help="Download the MIDI file",
                        use_container_width=True
                    )

                with dl3:
                    st.download_button(
                        label="🎵 Download Audio File",
                        data=results_audio,
                        file_name=f"{st.session_state['file_name']}.wav",
                        mime="audio/wav",
                        help="Download the audio file",
                        use_container_width=True
                    )

            except FileNotFoundError:
                st.error("❌ Fluidsynth not found. Please install it.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
