# 🎼 PicToMusic: Sheet Music to MIDI Converter

PicToMusic is an advanced computer vision application that transforms sheet music into playable MIDI files. Using state-of-the-art image processing and deep learning techniques, it detects and interprets musical notation from both digital images and camera captures.

## 🎯 Project Overview

1. **Optical Music Recognition (OMR)**
   - Advanced image processing for staff line and note detection
   - Robust handling of various sheet music formats and qualities
   - Real-time processing capabilities

2. **Musical Symbol Recognition**
   - CRNN (Convolutional Recurrent Neural Network) trained on 40,000+ music sheets
   - Accurate detection of notes, clefs, time signatures, and other musical symbols
   - Sophisticated handling of musical notation complexities

3. **Digital Music Generation**
   - Conversion to ABC notation format
   - MIDI file generation for playback
   - Support for various musical instruments and styles

## 🔬 Technical Implementation

### Image Processing Pipeline

```
┌─────────────────┐
│   Sheet Music   │
│     Image       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌───────────────────┐
│  Preprocessing  │      │   Image Quality   │
│  • Grayscale    ├─────►│   Enhancement     │
│  • Thresholding │      │   • Noise removal │
└────────┬────────┘      │   • Contrast adj. │
         │               └─────────┬─────────┘
         ▼                         │
    ┌────┴─────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│        Parallel Processing          │
│                                     │
│    ┌───────────┐      ┌──────────┐  │
│    │Staff Line │      │  Note    │  │
│    │Detection  │      │Detection │  │
│    └─────┬─────┘      └────┬─────┘  │
│          │                 │        │
└──────────┼─────────────────┼────────┘
           │                 │
           └──────┐     ┌────┘
                  │     │
                  ▼     ▼
         ┌─────────────────────┐
         │Symbol Segmentation  │
         │  • Position Data    │
         │  • Relative Spacing │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   CRNN Model        │
         │ ┌───────────────┐   │
         │ │ CNN Feature   │   │
         │ │ Extraction    │   │
         │ └───────┬───────┘   │
         │         ▼           │
         │ ┌───────────────┐   │
         │ │ LSTM Sequence │   │
         │ │ Learning      │   │
         │ └───────────────┘   │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   ABC Notation      │
         │   Generation        │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   MIDI Output       │
         │   Generation        │
         └─────────────────────┘
```

### Core Components

#### Optical Music Recognition (OMR)

1. **Image Preprocessing**
   ```python
   def improcess(image):
       # Convert to grayscale
       # Invert colors
       # Apply adaptive thresholding
   ```

2. **Staff Line Detection**
   ```python
   def find_staff_lines(image):
       # Detect horizontal lines
       # Group into staff systems
       # Extract staff properties
   ```

3. **Note Detection**
   ```python
   def find_notes(staff_lines):
       # Remove staff lines
       # Detect note components
       # Group related elements
   ```

#### Musical Symbol Recognition

(IN PROGRESS)

#### Digital Music Generation

(IN PROGRESS)

### Data Structures

```python
@dataclass
class StaffLine:
    index: int
    contour: np.ndarray
    bounds: Tuple[int, int, int, int]
    notes: List[Note]
    key: Optional[Key]

@dataclass
class Note:
    index: int
    relative_index: int
    line_index: int
    contour: np.ndarray
    bounds: Tuple[int, int, int, int]
    relative_position: Tuple[int, int]
    absolute_position: Tuple[int, int]
    label: Optional[str]
```

## 🛠️ Current Implementation Status

### Completed Features
- ✅ Basic image preprocessing and enhancement
- ✅ Staff line detection and segmentation
- ✅ Note component detection and grouping
- ✅ Interactive web interface with Streamlit
- ✅ Real-time image processing visualization
- ✅ Configurable processing parameters

### In Development
- 🔄 CRNN model integration for symbol recognition
- 🔄 ABC notation converter
- 🔄 MIDI generation system
- 🔄 Note classification and pitch detection
- 🔄 Time signature and rhythm analysis

### Future Enhancements
- 📋 Support for complex musical notations
- 📋 Real-time audio preview
- 📋 Mobile application development
- 📋 Batch processing capabilities
- 📋 Cloud-based processing option

## 🔧 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/AugustinMORVAL/PicToMusic
cd PicToMusic
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the web interface:
```bash
streamlit run app.py
```

2. Upload or capture sheet music:
   - Support for PNG, JPG, JPEG formats
   - Real-time camera capture available
   - Automatic image enhancement

3. Configure processing parameters:
   - Image resolution
   - Staff line detection sensitivity
   - Note detection parameters
   - Overlap threshold for component grouping

4. Process and generate output:
   - Visual feedback of detection results
   - ABC notation preview
   - MIDI file download

## 🔍 Technical Details

### Image Processing Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| Image Resolution | Max dimension | 1200px | 800-2000px |
| Staff Line Dilation | Line detection sensitivity | 3 | 1-10 |
| Note Dilation | Note detection sensitivity | 2 | 1-10 |
| Min Staff Area | Minimum staff line size | 10000 | 1000-20000 |
| Min Note Area | Minimum note size | 50 | 10-1000 |
| Overlap Threshold | Component grouping threshold | 0.5 | 0.1-0.9 |

### CRNN Model Architecture

- **Input**: Preprocessed image segments
- **Backbone**: ResNet-based feature extraction
- **Sequence Learning**: Bi-directional LSTM
- **Output**: Musical symbol classification
- **Training Data**: 40,000+ annotated sheet music samples

## 📚 Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Music21 Documentation](http://web.mit.edu/music21/doc/)
- [ABC Notation Guide](http://abcnotation.com/wiki/abc:standard)
- [MIDI File Format Specification](https://www.midi.org/specifications)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
