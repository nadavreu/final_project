# Real-Time Heart Rate Measurement from Video

This project measures heart rate using video input through photoplethysmography (PPG) techniques. It can process both live camera feeds and pre-recorded videos.

## Features
- Real-time video capture from webcam or video files
- Face detection and ROI tracking
- Advanced heart rate measurement using PPG principles:
  - Adaptive signal quality assessment
  - Dynamic confidence metrics
  - Robust noise filtering
  - Automatic signal quality indicators
- Real-time GUI display with visualization:
  - Live video feed with ROI overlay
  - Signal quality indicators
  - Confidence-based heart rate display
  - Real-time signal plots
- Signal processing optimizations using C++ (via DLLs)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── src/
│   ├── video/              # Video capture and processing
│   ├── face_detection/     # Face detection and ROI tracking
│   ├── signal_processing/  # Signal processing and heart rate calculation
│   ├── gui/               # GUI implementation
│   └── cpp/               # C++ optimized implementations
├── tests/                 # Unit tests
├── data/                  # Sample data and resources
└── requirements.txt       # Project dependencies
```

## Usage

Run the main application:
```bash
python src/main.py
```

## Technical Details

### Signal Processing Pipeline

The heart rate measurement pipeline includes several sophisticated components:

1. **Preprocessing**:
   - ROI extraction and tracking
   - Green channel isolation
   - CLAHE enhancement
   - Adaptive baseline removal

2. **Signal Enhancement**:
   - Dynamic noise filtering
   - Notch filtering for power line interference
   - Bandpass filtering for heart rate frequency range
   - Independent Component Analysis (ICA)

3. **Heart Rate Calculation**:
   - FFT-based frequency analysis
   - Peak detection with quadratic interpolation
   - Confidence metric calculation
   - Signal quality assessment

4. **Quality Metrics**:
   - SNR-based quality assessment
   - Peak prominence analysis
   - Temporal consistency checking
   - Adaptive thresholding

### Performance Features

- Reduced minimum sample requirement (30 samples)
- Adaptive smoothing based on signal quality
- Dynamic confidence thresholds
- Real-time signal quality indicators
- Robust to motion artifacts
- Automatic recovery from signal loss

## Development

The project uses Python for the main application logic and C++ (via DLLs) for performance-critical operations. Key components:

1. Video Input: OpenCV-based video capture
2. Face Detection: Haar Cascade classifier
3. Signal Processing: 
   - ROI tracking with quality metrics
   - Adaptive color space processing
   - Enhanced CLAHE preprocessing
   - Multi-stage filtering pipeline
   - Advanced frequency analysis
4. GUI: PyQt5-based interface with real-time visualization

## License
[Your chosen license] 