# Neuron Spike Identification

## Project Overview

This Python application, developed by Victoria Carlsten, Hao Chen, Claire Cropper, and Shi Gu, leverages various signal processing techniques to identify and analyze neuron spike events from neural recording data. It includes functionality for loading data from different file formats, filtering signals to extract relevant features, detecting spikes based on thresholds, and clustering spikes for further analysis.

## Software Modules Overview

### Main Entry and Data Loading
- **Functionality**: Handles file input through a GUI, supports `.mat`, `.abf`, and `.h5` formats.
- **Key Operations**: Loads and segments data according to the specified intervals.

### Signal Processing
- **Functionality**: Includes filtering operations for spike detection and local field potentials (LFPs).
- **Filters Used**:
  - Bandpass filters to isolate spikes.
  - Low-pass filters for LFP extraction.

### Spike Detection
- **Functionality**: Implements threshold-based spike detection and distinguishes individual spikes based on a defined delay.
- **Methods**:
  - Thresholding for initial spike detection.
  - Delay-based differentiation to ensure accurate spike isolation.

### Clustering and PCA
- **Functionality**: Applies PCA to reduce dimensionality and facilitate clustering.
- **Techniques**:
  - PCA for feature extraction.
  - K-means clustering for categorizing spike waveforms.
 
### Template Optimization
- **Functionality**: Refines spike templates by iteratively adjusting to the mean of the closest matching spikes.
- **Key Operations**:
  - Calculates the initial template as the mean of the cluster data.
  - Iterates to refine the template by comparing phase space trajectories of the spikes and the current template.
  - Uses interpolation and Euclidean distance to identify and converge to the closest spike waveform.

### Visualization
- **Functionality**: Provides extensive plotting functionalities to visualize raw signals, filtered data, spike events, and clustering results.
- **Components**:
  - Time series plots of raw and filtered signals.
  - Clustering output visualization.
### Software Flow Chart
![Software flow chart](/Documentation/Flowchart.png)

## Development and Build Tools

### Python: 
Version 3.8.1 - Main programming language used for all computations and processing.

### NumPy:
Essential library for numerical operations on large, multi-dimensional arrays and matrices.

### SciPy:
Used for scientific and technical computing, such as signal processing and optimization.

### Matplotlib:
Primary library for creating static, interactive, and animated visualizations in Python.

### scikit-learn:
Utilized for implementing machine learning algorithms, particularly PCA and k-means clustering.

### h5py:
Provides a Pythonic interface to the HDF5 binary data format.
    
### pyabf:
Enables reading of Axon Binary Format files used in electrophysiology.

### tkinter:
Standard GUI toolkit for Python, used to create the file selection interface.


## Installation Guide
This section provides detailed steps to prepare the environment and run the Neuron Spike Identification Tool on your system.
### Prerequisites

Before installing and running this application, ensure you have Python 3.8 or higher installed on your system. Python can be installed from [python.org](https://www.python.org/downloads/).
### Environment Setup
1. **Install Python**:
   Ensure that Python and pip (Python's package installer) are installed on your system. You can verify the installation by running the following commands in your command prompt or terminal:

   ```bash
   python --version
   pip --version
2. **Install Required Libraries**:
   Install all necessary Python libraries using pip. Run the following command in your command prompt or terminal:
   ```
   pip install -r requirements.txt
   
### Running the Software
After setting up your environment and downloading the project, you can run the software by executing the main script:
```
python spikesortingvtjason.py

