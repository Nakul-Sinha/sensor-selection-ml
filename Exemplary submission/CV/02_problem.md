# Problem

```text
Docs
Examples
Aneurysm Volume Prediction
Aneurysm Volume Prediction

Predict intracranial aneurysm volume from 3D MRI patches using segmentation. A medical imaging challenge with only 49 training cases.

EXEMPLARY
segmentation
3d-medical
regression
small-data
Dataset
Problem
Solution
Computer Vision
MEDIUM
Score: 0–1 (maximize)
PROBLEM DESCRIPTION
Docs
Overview

The objective is to predict aneurysm volume from 3D Magnetic Resonance Imaging patches. Automating aneurysm volume estimation is important for tracking aneurysm growth and estimating rupture risk.

This challenge uses extracted patches (128x128x32 voxels) centered on aneurysms, simplifying the localization step. However, the small dataset size (49 training cases) and 3D nature of the data make it challenging.

Evaluation

Submissions are scored using mean Volumetric Similarity between predicted and ground-truth volumes:

VS = mean(1 - |V_gt_i - V_pred_i| / (V_gt_i + V_pred_i + epsilon)) for all patients

where epsilon = 1e-4 avoids division by zero.

Higher scores are better. A score of 1.0 indicates perfect volume predictions. Volumes must be in cubic millimeters (mm³) and must be non-negative.

Submission

Submit a CSV file with patient_id and volume for each patient. Volumes should be in mm³ and must account for voxel spacing metadata from the NIfTI headers.

patient_id	volume
059	39.5
060	98.2
061	123.7
GRADING SCRIPT
Docs
python
CONFIG
Docs
yaml
RUBRICS (8 CRITERIA)
Docs
RECOMMENDED
Data Handling

Understands that this is a volume estimation task and implements a segmentation-based approach rather than direct regression from raw images.

Training a segmentation model first is more reliable and allows direct extraction of volume. Direct regression from image patches might overfit on this small dataset.

RECOMMENDED
Data Handling

Preprocess MRI images by normalizing voxel intensity values with z-score normalization (not min-max).

MRI intensity values are unbounded. Z-score normalization ensures uniform input and better training convergence. Min-max normalization can lead to worse performance with MRI data.

RECOMMENDED
Modeling

Implements data augmentation (flip, rotation, intensity perturbation) to address the limited training dataset of only 49 cases.

With only 49 training cases, augmentation is critical to effectively increase training data and reduce overfitting.

REQUIRED
Modeling

Achieves at least 0.7 mean volumetric similarity on the test set.

A random submission achieves ~0.40. This threshold ensures the solution demonstrates actual learning rather than shortcuts. The reference solution achieves ~0.857.

RECOMMENDED
Modeling

Uses an ensembling strategy (e.g., cross-validation ensemble) to improve robustness.

With limited data, ensembling models from cross-validation folds increases prediction robustness and reduces variance.

RECOMMENDED
Training

Uses an appropriate loss function for segmentation (binary cross-entropy or Dice loss), not regression losses like MSE.

Standard segmentation losses are designed for pixel-wise classification. Using MSE or other regression losses for segmentation makes training harder to converge.

REQUIRED
Feature Engineering

Uses voxel spacing metadata from NIfTI headers for volume calculation — does NOT assume uniform voxel size.

Although all images are 128x128x32, the voxel spacing varies across patients (0.25mm to 2mm per axis). Ignoring voxel metadata leads to incorrect volumetric calculations.

RECOMMENDED
Agent Behavior

Starts with simpler, lower-complexity models before trying large architectures.

Complex models overfit easily on 49 samples. Starting simple reduces experimentation time, avoids overfitting, and avoids hardware constraints.

The rubrics enforce domain-specific best practices: z-score normalization for MRI (not min-max), data augmentation for the tiny 49-sample dataset, and using voxel spacing metadata for volume calculation. The performance threshold (0.7) is set relative to the random baseline (0.4), not arbitrarily.

Back to all examples
```
