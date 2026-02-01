# Photometric Classification of Tidal Disruption Events for LSST

## Introduction
This repository contains a machine learning pipeline designed to identify Tidal Disruption Events (TDEs) within the LSST (Legacy Survey of 
Space and Time) data stream. TDEs are rare, high-energy transients occurring when a star is disrupted by a supermassive black hole. 
Distinguishing them from the vast background of supernovae and active galactic nuclei (AGN) requires a classification strategy that 
leverages specific astrophysical signatures- namely, their unique color evolution, thermal stability, and power-law decay rate.

Our approach utilizes a hybrid "Mixture of Experts" ensemble that combines gradient boosting (CatBoost) with non-linear support models 
(MLP and K-Nearest Neighbors), achieving high precision.

---
## Repository Structure

```text
.
├── archives/                   # Directory for final contest submission files
├── datasets/                   # Directory for raw light curves and processed feature caches
├── models/                     # Directory for saving trained models (.pkl) and thresholds
├── results/                    # Output directory for prediction CSVs
├── output.txt                  # Terminal output from running train test commands
├── requirements.txt            # Required library installation (scikit-learn, extinction, catboost...)
├── config.py                   # Global configuration (paths, filter wavelengths, seeds)
├── main.py                     # Entry point for the pipeline CLI
└── src/
    ├── data_loader.py          # Data ingestion and preprocessing logic
    ├── features.py             # Feature extraction (Gaussian Processes, Physics fitting)
    └── machine_learning/       # Core ML logic
        ├── model_factory.py    # Hybrid Ensemble Classifier architecture definition
        ├── train.py            # Logic for cross-validation and model training
        ├── predict.py          # Logic for loading models and generating submissions
        ├── tune.py             # Hyperparameter optimization scripts
        └── experimental.py     # Experimental architectures (not used in final model)
```
---

## Installation & Usage

Requires Python 3.12-3.13 to be installed. Install the required dependencies:

    `pip install -r requirements.txt`

*We ran into issues with catboost not working on Python 3.14.*

### Running the Pipeline

The pipeline is controlled via main.py using command-line arguments.

*--Train* : Train the Model: This will load the training data, extract features (if not cached), perform stratified cross-validation, 
            and save the final production model to the models/ directory.

    `python main.py --train`

*--predict* : Generate Predictions: This loads the trained model from models/ and generates a submission file for the test set in results/.


    `python main.py --predict`

*--tune* : Tune the hyperparameters: Performs trials in optimizing parameters that are used in training the model, runs tune.py from main.py.

    `python main.py --tune`

To run the full pipeline with its current configuration, use:

    `python main.py --train --predict`

To run the hyperparameter tuning for a # of trials, use:
    `python main.py --tune --trials #`

---
## Methodology

We use a feature-based classification approach rather than operating on just the provided data. Because LSST light curves are sparse and irregularly sampled, we first model every object using a **2-Dimensional Gaussian Process (GP)**. This GP allows us to interpolate the light curve in both time and wavelength, providing a representation that is more effective for model learning.

From this model, we extract three main components:

* **Temporal Morphology:** We calculate Rise Time, Fade Time, and Full-Width Half-Max (FWHM) to characterize the event's geometry, specifically targeting the "fast rise, slow decay" which seems typical of TDEs.
* **Physics:** We fit the light curve residuals against known physical models, specifically the standard TDE power-law decay ($L \propto t^{-5/3}$) and the "fireball" rise model ($L \propto t^2$). The quality of these fits (Chi-Squared error) serves as a primary discriminator.
* **Thermodynamics & Color:** We extract pre-peak and post-peak* color gradients. Unlike supernovae which cool rapidly (redden), TDEs typically maintain stable, hot blackbody temperatures. We quantify this using $g-r$ color stability and the "Blue Energy Fraction" (ratio of UV/Blue flux to total flux).

## Machine Learning Model Overview

We apply a Hybrid Ensemble Classifier designed to balance sensitivity with robustness. The final prediction is a weighted average of three distinct architectural components:

* (48%) **Base Learner :** A CatBoost (Gradient Boost Decision Tree) model trained on the full feature set. <br>
* (32%) **Domain Experts :** Two specialized CatBoost models restricted to specific feature subsets (one for "Morphology"  and one for "Physics" characteristics). This prevents any one model from overfitting to noise when meaningful signals are too weak. <br>
* (20%) **Manifold Support :** A Multi-Layer Perceptron (Neural Network) and K-Nearest Neighbors classifier. These non-tree-based models help identify TDE candidates that lie on the correct manifold in feature space but might be missed by decision boundaries.<br>

---
## Implementation Details

__Important Note on Reproducibility__ :  When running this code it is important to note that the library we use for creating 
the gaussian process for feature extraction has a floating-point variation between CPU architectures. This, surprisingly, is 
enough to have a noticable effect on the model performance between computers with different CPUs.\
(Because some of the values we are dealing with for the features we use are so small already.)\
The specific culprit is `sklearn.gaussian_process.GaussianProcessRegressor`.\
<br>
In order to replicate our winning submission, you will need to use the existing processed data provided in `datasets/processed_(training/testing)_data.csv`, otherwise the output will likely not be the same since the program will have to 
recreate that data using the gaussian process library.

<br>
The classification model is `EnsembleClassifier` implemented in src/machine_learning/model_factory.py. 
It integrates:

CatBoost: Utilized for its robust handling of categorical data and superior performance on tabular physics data.

Scikit-Learn: Provides the MLP (Neural Network) and KNN implementations, as well as the pipeline infrastructure for 
scaling and imputation.

### Physics-Based Feature Engineering

* **Redshift Correction:** All temporal features (Rise Time, Fade Time, FWHM) are corrected for time dilation ($$t_{\text{rest}} = t_{\text{obs}} / (1+z)$$). Redshift is also used to derive absolute magnitude proxies.
* **Uncertainty Handling:** Flux uncertainties are incorporated directly into the Gaussian Process Kernel (Matern 3/2). The noise level ($\alpha$) of the GP is set to the square of the normalized flux error, ensuring that noisy data points have minimal influence on derived features.

### Feature Importance

The table below lists the most important features in the final classifier. The dominance of physics-based metrics (Template Matching, Power Law Error) over simple shape metrics shows the model is learning the physical signature of tidal disruption.

| Rank | Feature | Description |
| :--- | :--- | :--- |
| 1 | `template_chisq_tde` | Goodness-of-fit against a normalized TDE shape template. |
| 2 | `negative_flux_fraction` | Robust noise metric, distinguishes real transients from artifacts. |
| 3 | `duty_cycle` | Percentage of survey time the object was active (distinguishes AGN). |
| 4 | `tde_power_law_error` | Raw error of the $t^{-5/3}$ power law fit. |
| 5 | `log_tde_error` | Log-space error of the $t^{-5/3}$ power law fit. |
| 6 | `robust_duration` | Time span between 10th and 90th flux percentiles. |
| 7 | `percentile_ratio_80_max` | Shape metric identifying "plateaus" vs. "peaks". |
| 8 | `ls_wave` | Gaussian Process spectral coherence length scale. |
| 9 | `total_radiated_energy` | Integrated bolometric luminosity proxy. |
| 10 | `compactness` | Ratio of integrated flux area to peak flux (distinguishes blocky vs. peaked shapes). |

### Handling Imbalance

* **Stratified Cross-Validation:** Ensuring representative distributions of TDEs in every training fold.
* **Dynamic Class Weighting:** The `scale_pos_weight` parameter is calculated dynamically for each fold ($N_{negative} / N_{positive}$) to penalize false negatives.

---

## References

* Bhardwaj, K., et al. (2025). A photometric classifier for tidal disruption events in Rubin LSST. *Astronomy & Astrophysics*.
* van Velzen, S., et al. (2021). Optical-Ultraviolet Tidal Disruption Events. *Space Science Reviews*.
* Gezari, S. (2021). Tidal Disruption Events. *Annual Review of Astronomy and Astrophysics*.
