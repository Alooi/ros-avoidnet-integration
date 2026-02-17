from torchvision.transforms import Resize, Normalize, ToTensor, InterpolationMode, Grayscale
from torchvision.transforms.functional import to_pil_image
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SUIM_old(Dataset):
    def __init__(self, path, grid_size=32):
        self.path = path
        self.images_path = os.path.join(self.path, "images")
        self.masks_path = os.path.join(
            self.path, f"masks_grided_{grid_size}_threshold_0.5"
        )
        self.images_list = os.listdir(self.images_path)
        self.masks_list = os.listdir(self.masks_path)
        self.transform = transforms.Compose(
            [
                Resize([128, 128], interpolation=InterpolationMode.BILINEAR),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        mask_name = image_name.split(".")[0] + ".bmp"
        image_path = os.path.join(self.images_path, image_name)
        mask_path = os.path.join(self.masks_path, mask_name)
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Apply transforms to convert images and masks to tensors
        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask


class SUIM(Dataset):
    def __init__(self, path, grid_size=32):
        self.path = path
        self.images_path = os.path.join(self.path, "images")
        self.masks_path = os.path.join(
            self.path, f"masks_grided_{grid_size}_threshold_0.5"
        )
        self.images_list = os.listdir(self.images_path)
        self.masks_list = os.listdir(self.masks_path)
        self.transform = transforms.Compose(
            [
                Resize([155, 155], interpolation=InterpolationMode.BILINEAR),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        mask_name = image_name.split(".")[0] + ".bmp"
        image_path = os.path.join(self.images_path, image_name)
        mask_path = os.path.join(self.masks_path, mask_name)
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Apply transforms to convert images and masks to tensors
        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask


class SUIM_grayscale(Dataset):
    def __init__(self, path, grid_size=32):
        if path:
            self.path = path
            self.images_path = os.path.join(self.path, "images")
            self.masks_path = os.path.join(
                self.path, f"masks_grided_{grid_size}_threshold_0.5"
            )
            self.images_list = os.listdir(self.images_path)
            self.masks_list = os.listdir(self.masks_path)
        self.transform = transforms.Compose(
            [
                Resize([155, 155], interpolation=InterpolationMode.BILINEAR),
                Grayscale(num_output_channels=1),
                ToTensor(),
                Normalize(
                    mean=[0.5],
                    std=[0.5],
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        mask_name = image_name.split(".")[0] + ".bmp"
        image_path = os.path.join(self.images_path, image_name)
        mask_path = os.path.join(self.masks_path, mask_name)
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Apply transforms to convert images and masks to tensors
        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask

    def get_transform(self):
        return self.transform

    def get_mask_transform(self):
        return self.mask_transform
