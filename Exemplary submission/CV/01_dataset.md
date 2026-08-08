# Dataset

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
DATASET DESCRIPTION
Docs
Intracranial Aneurysm MRI Dataset
Overview

The original dataset "Royal Brisbane TOFMRA Intracranial Aneurysm Database" contains 63 patients with MRI scans showing intracranial aneurysms (85 total aneurysms). The raw dataset is over 25 GB of high-resolution 3D MRI volumes.

Diagnosing aneurysms requires high-resolution imaging because aneurysms vary in size — from very small (<1mm) to large (>5mm) in diameter. This causes a very imbalanced target-to-background ratio in brain MRI angiography scans.

The challenge dataset is modified: it includes only extracted patches (128x128x32 voxels) centered on aneurysms from the 3D volumes. The voxel spacing is anisotropic and ranges from 0.25mm to 1-2mm. Voxel spacing metadata can be extracted from the NIfTI headers.

The dataset is licensed under CC0.

Chloe M. de Nys et al. (2024). Royal Brisbane TOFMRA Intracranial Aneurysm Database. OpenNeuro. doi:10.18112/openneuro.ds005096.v1.0.3

File Structure
public/train/ — 49 training image patches (.nii.gz)
public/train_labels/ — 49 corresponding segmentation masks (.nii.gz)
public/train.csv — Patient IDs mapped to ground-truth volumes (mm³)
public/test/ — 10 test image patches (.nii.gz)
public/sample_submission.csv — Example submission format
private/answers.csv — Ground-truth test volumes
Data Characteristics
Image format: NIfTI (.nii.gz), 3D volumes (128x128x32 voxels)
Voxel spacing: Anisotropic, varies per patient (0.25mm to 2mm per axis)
Training set: 49 patients with paired images and segmentation masks
Test set: 10 patients (images only, no masks)
Small dataset: Only 49 training cases — data augmentation is critical
Important: Voxel spacing is NOT uniform across patients. Volume calculations must use per-patient voxel metadata from NIfTI headers.

The dataset description highlights two critical details: anisotropic voxel spacing (0.25mm to 2mm) and that patches are pre-centered on aneurysms. These shape the entire modeling approach — participants must use voxel metadata for volume calculation, and the centering means segmentation is feasible even with a small model.

FILE STRUCTURE
Docs
Files
PREPARE SCRIPT
Docs
python

The prepare script handles real medical imaging complexity: reorienting NIfTI volumes to standard LPS orientation, finding connected components, and extracting centered patches. Volume is computed in physical units (mm³) using image metadata — not raw voxel counts.

Back to all examples
```
