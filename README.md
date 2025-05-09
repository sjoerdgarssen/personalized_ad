# Dummy Experiment for Personalized Anomaly Detection

This repository contains a dummy implementation for personalized anomaly detection using sensor data as used in the corresponding article.

## Features

- **Dummy Data Generation**: Simulates sensor data for multiple patients.
- **Anomaly Detection Algorithms**: Implements Local Outlier Factor and Isolation Forest for anomaly detection.
- **SHAP Analysis**: Calculates SHAP values to explain model predictions for deteriorated patients.
- **REWS Calculation**: Computes REWS based on sensor data.

## Folder Structure

- `utils/`: Contains utility functions for training models, applying algorithms, and generating dummy data.
- `dummy_experiment.py`: Main script to run the dummy experiment.

## Citation

If you use this project in your research, please cite the original paper.

## Requirements

- Python 3.8 or higher
- Used Python libraries:
    - numpy==1.23.5
    - pandas==1.4.4
    - matplotlib==3.6.3
    - seaborn==0.12.0
    - scikit-learn==1.2.0
    - shap==0.44.1

