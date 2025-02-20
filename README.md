# Collusion Detection Demo

This Streamlit application provides a demonstration of collusion detection techniques using Z-Scoring and Hierarchical Agglomerative Clustering (HAC). The app is designed to identify potential colluding entities by analyzing relational data and visualizing clustering results.

## Features

- **HAC (Hierarchical Agglomerative Clustering):**
  - Groups similar objects using a hierarchical structure.
  - Visualizes clustering using dendrograms to identify colluding entities.
  - Allows selection of linkage criteria and thresholds for optimal clustering.

- **Z-Scoring:**
  - Standardizes data points to detect anomalies or outliers.
  - Visualizes data using treemaps and bar charts for intuitive understanding.

## Installation

1. Clone the repository

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run Collusion_Detection_Demo.py
    ```

2. Navigate to the sidebar to select the desired analysis method (HAC or Z-Scoring).

3. For Z-Scoring:
   - Upload a text file containing absolute time and Levenshtein distances as well as the correlation value and z-scores for each relation dimension. See `example2.txt` for an example.
   - Adjust the thresholds using the sliders to filter colluding students.

4. For HAC:
   - Upload a text file the distances between all pairs of students. See `example1.txt` for an example.

5. Regarding preprocessing steps please refer to `preprocessing.py` for the HAC preprocessing and `preprocessing_abs.py` for the Z-Scoring preprocessing. Please adapt the preprocessing code to your own data.

## File Structure

- `Collusion_Detection_Demo.py`: Main entry point of the Streamlit application.
- `pages/`: Contains individual pages for different analysis methods (e.g., HAC and Z-Scoring).
- `utils/`: Utility functions used throughout the application.
- **Important: Add the absolute path to the top of each page (there already is a comment highlighting the position), otherwise the helper functions can not be found.**