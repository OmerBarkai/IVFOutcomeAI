# IVF Pregnancy Prediction using Microbiome and Inflammatory Markers

A machine learning framework for predicting IVF pregnancy outcomes based on vaginal microbiome composition and inflammatory cytokine profiles.

## Overview

This project implements a machine learning approach to predict pregnancy outcomes in IVF patients using data collected at three different timepoints during IVF treatment. The prediction model analyzes:

- **Vaginal microbiome composition**: Bacterial taxonomic abundance data
- **Inflammatory markers**: Cytokine concentration profiles
- **Combined feature sets**: Integrated analysis of both microbiome and cytokine data

The implementation uses Support Vector Classification with Leave-One-Out Cross-Validation and SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance in the outcome data.

## Features

- **Multi-timepoint analysis**: Evaluates data from three distinct sampling timepoints during IVF treatment
- **Multiple feature set evaluation**: Independent and combined analysis of bacterial and cytokine features
- **Advanced interpretability**: SHAP (SHapley Additive exPlanations) values for model interpretation
- **Comprehensive visualization**: Performance metrics, confusion matrices, and feature importance plots
- **Cross-validation**: Leave-One-Out Cross-Validation for robust performance assessment
- **Class imbalance handling**: SMOTE implementation for addressing unbalanced class distribution

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (for SMOTE)
- shap


## Usage

1. Prepare your data files:
   - Main microbiome and inflammation data CSV
   - IVF cycle type information

2. Update file paths in the script:
```python
MICROBIOME_PATH = 'path/to/miio_all.csv'
IVF_CYCLE_TYPE_PATH = 'path/to/MIIO_IVFcycleType.csv'
```

3. Run the analysis:
```bash
python ivf_prediction.py
```

4. Review generated visualizations:
   - Performance plots with accuracy and F1 scores
   - Confusion matrices
   - SHAP summary plots
   - Feature importance heatmaps

## Data Structure

The analysis expects three primary data files:

1. **Microbiome Data**: CSV containing bacterial abundance and cytokine measurements
   - Columns 1-20: Cytokine features
   - Columns 25-275: Bacterial features
   - Must include 'sample' and 'outcome' columns

2. **Shannon Index Data**: Tab-separated file with diversity measurements
   - Must include 'Shannon_index' and 'sample' columns

3. **IVF Cycle Type Data**: CSV with treatment information
   - Must include 'sample' and 'Cycle.IVF.Cryothaw' columns

## Configuration Parameters

The script includes several configurable parameters:

```python
# Define timepoints and feature sets for analysis
TIME_POINTS = ['1A', '2A', '3A']  # Three different sampling timepoints
FEATURE_SETS = {
    'Cytokines': slice(1, 21),      # Cytokine features (columns 1-20)
    'Bacteria': slice(25, 275),     # Bacterial features (columns 25-275)
    'Cytokines and Bacteria': None  # Combined feature set
}

# Analysis parameters
CLASS_THRESHOLD = 0.5           # Threshold for classification
OUTCOME_COLUMN = 'YesPreg'      # Target variable name
MIN_SAMPLE_THRESHOLD = 0.5      # Minimum required non-zero values in features
N_TOP_FEATURES = 10             # Number of top features to display in plots
```

## Output

The script generates various visualizations:

1. **Performance Plots**: Scatter plots showing prediction accuracy by IVF cycle type
2. **Confusion Matrices**: Normalized true/false positive/negative rates
3. **SHAP Summary Plots**: Feature importance and impact on model predictions
4. **Feature Importance Heatmaps**: Comparison across timepoints and feature sets
5. **F1 Score Heatmaps**: Performance metrics for each analysis combination

### License


## Citation

If you use this code in your research, please cite:

[TBD]

## Acknowledgments

This project uses the custom utility functions from the AniML_utils_Publishing module.
