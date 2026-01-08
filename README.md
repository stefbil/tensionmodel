# A Computational Model for Predicting Perceived Tension in Electronic Dance Music Build-ups

This feature-based model analyzes musical tension in Electronic Dance Music (EDM) build-up sections by operationalizing Gestalt principles through audio signal processing. The application extracts audio features to quantify systematic violations of perceptual grouping principles that create tension.

## Overview

In EDM, build-ups are structural elements designed to maximize tension before a drop. This project models this tension using three key dimensions:

1.  **Pitch Proximity Violation (Risers):** Measures continuous rising pitch using the YIN algorithm.
2.  **Timbral Similarity Violation (Brightness):** Tracks increases in spectral brightness (spectral centroid), often caused by high-pass filter sweeps.
3.  **Temporal Proximity Increase (Rhythmic Density):** Quantifies accelerating rhythmic elements (e.g., snare rolls) using Inverse Inter-Onset Intervals (IOI).

These features are normalized and combined into a **Composite Tension Index**.

## Features

-   **Single Track Analysis:** Upload specific build-up sections for detailed analysis.
-   **Interactive Visualizations:** Zoom and hover over plots for:
    -   Waveform RMS
    -   Combined Tension Curve
    -   Normalized Feature Contributions (0-1)
    -   Raw Feature Values (Hz, IOI)
    -   Correlation Heatmaps
-   **Batch Analysis:** Overlay tension curves from up to 12 tracks to compare build-up profiles.
-   **Customizable Parameters:** Adjust analysis settings like FFT size, hop length, smoothing windows, and feature weights via the sidebar.
-   **Data Export:** Download computed tension data and features as CSV files.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/stefbil/tensionmodel
    cd tensionmodel
    ```

2.  **Install dependencies:**
    Ensure you have Python installed (preferably 3.10 or later). Install the required packages using pip in a new environment:

    ```bash
    pip install -r requirements.txt
    ```


## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app3.py
    ```

2.  **Analyze Audio:**
    -   Open the provided local URL in your web browser.
    -   **Single File:** Use the first file uploader to analyze a single EDM build-up (8-16 bars recommended).
    -   **Batch Analysis:** Scroll down to the "Overlay of Tension Curves" section to upload multiple files for comparison.

3.  **Adjust Settings:** Use the sidebar to fine-tune the audio analysis (e.g., Sample Rate, STFT window) and the weighting of the tension components (Pitch vs. Centroid vs. Density).


## Author

**Stefanos Biliousis**
Aalborg University, Copenhagen
Sound and Music Perception and Cognition Semester Project
