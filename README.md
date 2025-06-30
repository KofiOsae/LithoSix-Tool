# Enhanced AI-Based LithoSix Tool

This is a Streamlit web application integrating multiple lithography process analysis components with AI enhancements:

- **DOE Manager:** Define Design of Experiments and auto-suggest new DOEs based on prior process capability indices (Cp, Cpk).
- **SEM Analyzer:** Upload SEM images for edge detection and quality scoring using a CNN model.
- **Six Sigma Stats:** Calculate process capability indices (Cp, Cpk), perform ANOVA tests on input data.
- **Trend Dashboard:** Visualize time-series data with SPC charts and detect anomalies.
- **Data Export:** Export DOE history data as CSV for external analysis.

## Features

- Interactive UI to add and manage DOE experiments.
- Image processing with OpenCV and AI classification using TensorFlow.
- Statistical calculations and tests for process monitoring.
- Dynamic visualization with Plotly.
- Session-state management for data persistence.

## Requirements

Python 3.8+ and the libraries listed in [`requirements.txt`](./requirements.txt).

## Installation

```bash
git clone <repo-url>
cd lithosix-tool
pip install -r requirements.txt
Usage
Select the desired tool from the sidebar.

Input your data, upload images, and interact with each module.

Export DOE data for external tools like JMP or Minitab.

Notes
The CNN model in the SEM Analyzer is a placeholder; replace with your trained weights for accurate classification.

AI suggestions in DOE Manager use simple heuristics and can be improved with real machine learning models.

SPC and anomaly detection use Isolation Forest as a baseline method.

Contributing
Contributions and improvements are welcome! Please submit issues or pull requests.
