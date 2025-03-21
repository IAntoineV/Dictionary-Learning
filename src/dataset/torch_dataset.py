import os
from enum import Enum
from PIL import Image
import torch
from torchvision import transforms

vanilla_transform = transforms.Compose(
    [
        transforms.ToTensor()  # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    ]
)

class NormalizationType(Enum):
    NONE = "none"
    L2 = "l2"
    ZSCORE = "zscore"


class FlattenedPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        directory,
        patch_size=(12, 12),
        patches_per_image=5,
        transform=vanilla_transform,
        norm_type: NormalizationType=NormalizationType.L2
    ):
        self.image_paths = [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if file.endswith(".jpg")
        ]
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transform
        self.norm_type = norm_type

    def __len__(self):
        return self.patches_per_image * len(self.image_paths)

    def __getitem__(self, idx):
        image_idx = idx // self.patches_per_image

        img = Image.open(self.image_paths[image_idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        patch = self._extract_random_patch(img)

        patch_flat = patch.flatten()

        patch_flat = self._normalize_flat_patch(patch_flat)

        return patch_flat

    def _extract_random_patch(self, image):
        _, img_h, img_w = image.shape  # [C, H, W]
        patch_h, patch_w = self.patch_size

        if img_h < patch_h or img_w < patch_w:
            raise ValueError(
                f"Image size ({img_h}, {img_w}) is smaller than patch size {self.patch_size}!"
            )

        y = torch.randint(0, img_h - patch_h + 1, (1,)).item()
        x = torch.randint(0, img_w - patch_w + 1, (1,)).item()

        return image[:, y : y + patch_h, x : x + patch_w]

    def _normalize_flat_patch(self, patch_flat):
        if self.norm_type == NormalizationType.L2:
            l2_norm = torch.norm(patch_flat, p=2)
            if l2_norm > 0:
                patch_flat = patch_flat / l2_norm

        elif self.norm_type == NormalizationType.ZSCORE:
            mean = patch_flat.mean()
            std = patch_flat.std()
            if std > 0:
                patch_flat = (patch_flat - mean) / std

        return patch_flat


class PatchAndFlatenDataset(FlattenedPatchDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, idx):
        image_idx = idx // self.patches_per_image

        img = Image.open(self.image_paths[image_idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        patch = self._extract_random_patch(img)

        patch_flat = patch.flatten()

        patch_flat = self._normalize_flat_patch(patch_flat)

        return patch, patch_flat


class PatchDataset(FlattenedPatchDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.scaler = StandardScaler()
    
    def __getitem__(self, idx):
        image_idx = idx // self.patches_per_image

        img = Image.open(self.image_paths[image_idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        patch = self._extract_random_patch(img)
        
        # patch = torch.tensor(self.scaler.fit_transform(patch.numpy()))
        return patch
