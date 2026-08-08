# Aneurysm Volume Prediction

> Predict intracranial aneurysm volume from 3D MRI patches using segmentation. A medical imaging challenge with only 49 training cases.

**Status:** EXEMPLARY
**Tags:** segmentation · 3d-medical · regression · small-data
**Domain:** Computer Vision **Difficulty:** MEDIUM **Score:** 0–1 (maximize)

Source: https://shipd.ai/quests/eris/docs/examples/aneurysm-volume-prediction

---

## Dataset

### Intracranial Aneurysm MRI Dataset

#### Overview

The original dataset "Royal Brisbane TOFMRA Intracranial Aneurysm Database" contains 63 patients with MRI scans showing intracranial aneurysms (85 total aneurysms). The raw dataset is over 25 GB of high-resolution 3D MRI volumes.

Diagnosing aneurysms requires high-resolution imaging because aneurysms vary in size — from very small (<1mm) to large (>5mm) in diameter. This causes a very imbalanced target-to-background ratio in brain MRI angiography scans.

The challenge dataset is modified: it includes only extracted patches (128x128x32 voxels) centered on aneurysms from the 3D volumes. The voxel spacing is anisotropic and ranges from 0.25mm to 1-2mm. Voxel spacing metadata can be extracted from the NIfTI headers.

The dataset is licensed under CC0.

> Chloe M. de Nys et al. (2024). Royal Brisbane TOFMRA Intracranial Aneurysm Database. OpenNeuro. doi:10.18112/openneuro.ds005096.v1.0.3

#### File Structure

- `public/train/` — 49 training image patches (.nii.gz)
- `public/train_labels/` — 49 corresponding segmentation masks (.nii.gz)
- `public/train.csv` — Patient IDs mapped to ground-truth volumes (mm³)
- `public/test/` — 10 test image patches (.nii.gz)
- `public/sample_submission.csv` — Example submission format
- `private/answers.csv` — Ground-truth test volumes

#### Data Characteristics

- **Image format:** NIfTI (.nii.gz), 3D volumes (128x128x32 voxels)
- **Voxel spacing:** Anisotropic, varies per patient (0.25mm to 2mm per axis)
- **Training set:** 49 patients with paired images and segmentation masks
- **Test set:** 10 patients (images only, no masks)
- **Small dataset:** Only 49 training cases — data augmentation is critical
- **Important:** Voxel spacing is NOT uniform across patients. Volume calculations must use per-patient voxel metadata from NIfTI headers.

> The dataset description highlights two critical details: anisotropic voxel spacing (0.25mm to 2mm) and that patches are pre-centered on aneurysms. These shape the entire modeling approach — participants must use voxel metadata for volume calculation, and the centering means segmentation is feasible even with a small model.

### Prepare Script

```python
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label as cc_label, center_of_mass
import csv
import re

PATCH_SIZE = (128, 128, 32)
TARGET_ORIENTATION = ("L", "P", "S")


def prepare(raw: Path, public: Path, private: Path):
    """Prepare aneurysm patches from raw MRI data.

    Steps:
    1. Collect paired image/label NIfTI files from raw dataset
    2. Reorient all volumes to standard LPS orientation
    3. For each patient, find the largest aneurysm component
    4. Extract 128x128x32 patches centered on the aneurysm
    5. Compute volume in mm3 using voxel spacing metadata
    6. Split into train (49 patients) / test (10 patients)
    """
    images, labels_map = collect_image_label_pairs(raw)

    # Manual test split for reproducibility
    MANUAL_TEST_KEYS = [
        "sub-059_ses-20210208", "sub-060_ses-20160205",
        "sub-061_ses-20210427", "sub-062_ses-20210623",
        "sub-063_ses-20210628", "sub-064_ses-20210418",
        "sub-065_ses-20190607", "sub-067_ses-20200207",
        "sub-069_ses-20170209", "sub-070_ses-20181227",
    ]

    keys = [k for k in sorted(images.keys())
            if k in labels_map and len(labels_map[k]) > 0]
    test_keys = set(MANUAL_TEST_KEYS)
    train_keys = set(k for k in keys if k not in test_keys)

    for d in ["train", "train_labels", "test"]:
        (public / d).mkdir(parents=True, exist_ok=True)
    (private / "test_labels").mkdir(parents=True, exist_ok=True)

    train_rows, test_rows = [], []

    for key in sorted(train_keys | test_keys):
        # Load and reorient MRI to standard LPS orientation
        img_nii = reorient_nifti(nib.load(str(images[key])))
        img = np.asanyarray(img_nii.dataobj)

        # Merge all aneurysm masks for this patient
        mask = merge_masks(labels_map[key], img.shape)

        # Find largest connected component center
        labeled, num = cc_label(mask > 0)
        center = center_of_mass(mask, labeled, [1])[0]

        # Extract centered patch and save
        img_patch, seg_patch, new_aff = extract_centered_patch(
            img, mask, center, PATCH_SIZE, img_nii.affine
        )

        patient_id = re.search(r"sub-(\d+)", key).group(1)
        is_test = key in test_keys
        save_dir = public / ("test" if is_test else "train")

        nib.save(
            nib.Nifti1Image(img_patch, new_aff),
            str(save_dir / f"{patient_id}.nii.gz"),
        )

        # Volume = voxel count x voxel volume (using NIfTI header)
        sx, sy, sz = img_nii.header.get_zooms()[:3]
        volume_mm3 = float((seg_patch > 0).sum() * sx * sy * sz)
        rows = test_rows if is_test else train_rows
        rows.append((patient_id, volume_mm3))

    write_csv(public / "train.csv", train_rows)
    write_csv(private / "answers.csv", test_rows)
```

> The prepare script handles real medical imaging complexity: reorienting NIfTI volumes to standard LPS orientation, finding connected components, and extracting centered patches. Volume is computed in physical units (mm³) using image metadata — not raw voxel counts.

---

## Problem

**Domain:** Computer Vision **Difficulty:** MEDIUM **Score:** 0–1 (maximize)

### Problem Description

#### Overview

The objective is to predict aneurysm volume from 3D Magnetic Resonance Imaging patches. Automating aneurysm volume estimation is important for tracking aneurysm growth and estimating rupture risk.

This challenge uses extracted patches (128x128x32 voxels) centered on aneurysms, simplifying the localization step. However, the small dataset size (49 training cases) and 3D nature of the data make it challenging.

#### Evaluation

Submissions are scored using **mean Volumetric Similarity** between predicted and ground-truth volumes:

```
VS = mean(1 - |V_gt_i - V_pred_i| / (V_gt_i + V_pred_i + epsilon)) for all patients
```

where epsilon = 1e-4 avoids division by zero.

**Higher scores are better.** A score of 1.0 indicates perfect volume predictions. Volumes must be in cubic millimeters (mm³) and must be non-negative.

#### Submission

Submit a CSV file with patient_id and volume for each patient. Volumes should be in mm³ and must account for voxel spacing metadata from the NIfTI headers.

| patient_id | volume |
|------------|--------|
| 059 | 39.5 |
| 060 | 98.2 |
| 061 | 123.7 |

### Grading Script

```python
import math
import pandas as pd


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    answers = answers[["patient_id", "volume"]].copy()
    submission = submission[["patient_id", "volume"]].copy()

    answers["patient_id"] = answers["patient_id"].astype(str).str.strip()
    submission["patient_id"] = submission["patient_id"].astype(str).str.strip()

    answers["volume"] = pd.to_numeric(answers["volume"], errors="coerce")
    submission["volume"] = pd.to_numeric(submission["volume"], errors="coerce")

    gt_map = dict(zip(answers["patient_id"], answers["volume"]))
    pred_map = dict(zip(submission["patient_id"], submission["volume"]))

    patient_ids = sorted(set(gt_map.keys()) | set(pred_map.keys()))

    eps = 1e-4
    scores = []

    for pid in patient_ids:
        g = gt_map.get(pid, None)
        p = pred_map.get(pid, None)

        if g is None or p is None or pd.isna(g) or pd.isna(p):
            raise ValueError(f"Missing or non-numeric volume for patient: {pid}")

        g = float(g)
        p = float(p)

        if (not math.isfinite(g)) or (not math.isfinite(p)) or g < 0 or p < 0:
            raise ValueError(f"Invalid volume for patient: {pid}")

        scores.append(1.0 - (abs(g - p) / (g + p + eps)))

    return float(sum(scores) / len(scores)) if scores else 0.0
```

### Config

```yaml
name: Aneurysm Volume Prediction
difficulty: medium
domain: vision

grade:
  direction: maximize
  minimum: 0
  maximum: 1
```

### Rubrics (8 criteria)

1. **[RECOMMENDED] Data Handling** — Understands that this is a volume estimation task and implements a segmentation-based approach rather than direct regression from raw images.
   *Training a segmentation model first is more reliable and allows direct extraction of volume. Direct regression from image patches might overfit on this small dataset.*

2. **[RECOMMENDED] Data Handling** — Preprocess MRI images by normalizing voxel intensity values with z-score normalization (not min-max).
   *MRI intensity values are unbounded. Z-score normalization ensures uniform input and better training convergence. Min-max normalization can lead to worse performance with MRI data.*

3. **[RECOMMENDED] Modeling** — Implements data augmentation (flip, rotation, intensity perturbation) to address the limited training dataset of only 49 cases.
   *With only 49 training cases, augmentation is critical to effectively increase training data and reduce overfitting.*

4. **[REQUIRED] Modeling** — Achieves at least 0.7 mean volumetric similarity on the test set.
   *A random submission achieves ~0.40. This threshold ensures the solution demonstrates actual learning rather than shortcuts. The reference solution achieves ~0.857.*

5. **[RECOMMENDED] Modeling** — Uses an ensembling strategy (e.g., cross-validation ensemble) to improve robustness.
   *With limited data, ensembling models from cross-validation folds increases prediction robustness and reduces variance.*

6. **[RECOMMENDED] Training** — Uses an appropriate loss function for segmentation (binary cross-entropy or Dice loss), not regression losses like MSE.
   *Standard segmentation losses are designed for pixel-wise classification. Using MSE or other regression losses for segmentation makes training harder to converge.*

7. **[REQUIRED] Feature Engineering** — Uses voxel spacing metadata from NIfTI headers for volume calculation — does NOT assume uniform voxel size.
   *Although all images are 128x128x32, the voxel spacing varies across patients (0.25mm to 2mm per axis). Ignoring voxel metadata leads to incorrect volumetric calculations.*

8. **[RECOMMENDED] Agent Behavior** — Starts with simpler, lower-complexity models before trying large architectures.
   *Complex models overfit easily on 49 samples. Starting simple reduces experimentation time, avoids overfitting, and avoids hardware constraints.*

> The rubrics enforce domain-specific best practices: z-score normalization for MRI (not min-max), data augmentation for the tiny 49-sample dataset, and using voxel spacing metadata for volume calculation. The performance threshold (0.7) is set relative to the random baseline (0.4), not arbitrarily.

---

## Solution

**Score: 0.86**

### Solution Code

```python
"""
3D UNet segmentation approach for aneurysm volume prediction.
Trains a segmentation model on MRI patches, then computes volume
from predicted masks using voxel spacing metadata.
"""
import os, glob, csv, sys
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Runtime paths
public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

# Configuration
training_images = public_dir / "train"
training_labels = public_dir / "train_labels"
testing_images = public_dir / "test"
checkpoint_path = submission_out.parent / "unet3d_best.pt"
submission_csv_path = submission_out
submission_out.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
UNET_BASE = 16
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NiftiPatchDataset(Dataset):
    """Training dataset with z-score normalization and augmentation."""

    def __init__(self, img_dir, seg_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
        self.seg_dir = seg_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.asarray(nib.load(self.img_paths[idx]).dataobj, dtype=np.float32)
        seg_path = os.path.join(self.seg_dir, os.path.basename(self.img_paths[idx]))
        seg = np.asarray(nib.load(seg_path).dataobj, dtype=np.float32)

        # Z-score normalization (appropriate for MRI, not min-max)
        img = (img - img.mean()) / (img.std() + 1e-8)
        seg = (seg > 0).astype(np.float32)

        img_t = torch.from_numpy(np.transpose(img, (2, 1, 0))).unsqueeze(0)
        seg_t = torch.from_numpy(np.transpose(seg, (2, 1, 0))).unsqueeze(0)

        # Data augmentation: random flips and rotations (critical for 49 samples)
        if torch.rand(()) < 0.5:
            img_t = torch.flip(img_t, dims=(3,))
            seg_t = torch.flip(seg_t, dims=(3,))
        if torch.rand(()) < 0.5:
            img_t = torch.flip(img_t, dims=(2,))
            seg_t = torch.flip(seg_t, dims=(2,))
        if torch.rand(()) < 0.5:
            k = int(torch.randint(0, 4, (1,)).item())
            img_t = torch.rot90(img_t, k, (2, 3))
            seg_t = torch.rot90(seg_t, k, (2, 3))

        return img_t, seg_t


# Lightweight 3D UNet (3 encoder stages - avoids overfitting on small dataset)
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch), nn.ReLU(True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = ConvBlock3D(1, base)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(base * 4, base * 4)

        self.up3 = nn.ConvTranspose3d(base * 4, base * 4, 2, stride=2)
        self.dec3 = ConvBlock3D(base * 8, base * 2)
        self.up2 = nn.ConvTranspose3d(base * 2, base * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base)
        self.up1 = nn.ConvTranspose3d(base, base, 2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)
        self.out_conv = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.bottleneck(self.pool(s3))
        x = self.dec3(torch.cat([self.up3(x), s3], 1))
        x = self.dec2(torch.cat([self.up2(x), s2], 1))
        x = self.dec1(torch.cat([self.up1(x), s1], 1))
        return self.out_conv(x)


# Training
dataset = NiftiPatchDataset(training_images, training_labels)
n_train = int(len(dataset) * 0.8)
train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])

model = UNet3D(UNET_BASE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
best_dice = -1.0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    for imgs, segs in DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True):
        optimizer.zero_grad()
        loss = criterion(model(imgs.to(DEVICE)), segs.to(DEVICE))
        loss.backward()
        optimizer.step()

    # Validate with Dice score
    model.eval()
    dice_sum = 0
    with torch.no_grad():
        for imgs, segs in DataLoader(val_ds, batch_size=BATCH_SIZE):
            probs = torch.sigmoid(model(imgs.to(DEVICE)))
            preds = (probs > 0.5).float().view(probs.size(0), -1)
            tgts = segs.to(DEVICE).view(segs.size(0), -1)
            inter = (preds * tgts).sum(1)
            dice_sum += ((2 * inter) / (preds.sum(1) + tgts.sum(1) + 1e-6)).mean()

    val_dice = dice_sum / max(len(val_ds) // BATCH_SIZE, 1)
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), checkpoint_path)
    if epoch % 25 == 0:
        print(f"Epoch {epoch}/{NUM_EPOCHS} | val dice {val_dice:.4f}")

print(f"Best val dice: {best_dice:.4f}")

# Inference: predict segmentation -> compute volume using voxel spacing
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()
volume_rows = []

for path in sorted(glob.glob(os.path.join(testing_images, "*.nii*"))):
    nii = nib.load(path)
    img = np.asarray(nii.dataobj, dtype=np.float32)
    img = (img - img.mean()) / (img.std() + 1e-8)
    img_t = torch.from_numpy(np.transpose(img, (2, 1, 0))).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = (torch.sigmoid(model(img_t.to(DEVICE))) >= THRESHOLD)
        pred = pred.squeeze().cpu().numpy()

    # Volume = voxel count x voxel volume (from NIfTI header metadata)
    sx, sy, sz = nii.header.get_zooms()[:3]
    volume_mm3 = float(np.transpose(pred, (2, 1, 0)).astype(np.uint8).sum() * sx * sy * sz)
    patient_id = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
    volume_rows.append((patient_id, volume_mm3))

with open(submission_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "volume"])
    writer.writerows(volume_rows)

print(f"Submission saved with {len(volume_rows)} predictions")
```

> The solution uses a lightweight 3D UNet (base=16) — a deliberate choice for the small dataset. Data augmentation via random flips and rotations effectively multiplies the training data. The volume calculation correctly multiplies voxel counts by per-patient voxel spacing from the NIfTI header, achieving 0.857 volumetric similarity.

### Output Files

- `solution.ipynb` — 15 KB
- `working/unet3d_best.pt` — 1.2 MB
- `submission.csv` — 1 KB
