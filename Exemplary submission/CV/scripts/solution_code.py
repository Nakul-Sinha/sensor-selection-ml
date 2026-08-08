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