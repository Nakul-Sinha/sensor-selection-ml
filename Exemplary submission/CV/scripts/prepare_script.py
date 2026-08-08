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