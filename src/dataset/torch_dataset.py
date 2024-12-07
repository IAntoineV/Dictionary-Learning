import os
from PIL import Image
import torch
from torchvision import transforms

vanilla_transform = transforms.Compose(
    [
        transforms.ToTensor()  # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    ]
)


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        directory,
        patch_size=(12, 12),
        patches_per_image=5,
        transform=vanilla_transform,
    ):
        self.image_paths = [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if file.endswith(".jpg")
        ]
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transform

    def __len__(self):
        return self.patches_per_image * len(self.image_paths)

    def __getitem__(self, idx):
        image_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        img = Image.open(self.image_paths[image_idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        patch = self._extract_random_patch(img)
        return patch

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
