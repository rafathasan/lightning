import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from lightning.pytorch import LightningDataModule
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BingRGB(LightningDataModule):
    def __init__(self, data_dir, batch_size, val_batch_size, num_workers):
        super(BingRGB, self).__init__()

        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        
    def prepare_data(self):
        # Nothing to prepare here
        pass
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_transforms = self.get_train_transforms()
            val_transforms = self.get_test_transforms()
            
            self.train_dataset = self.BingRGBDataset(os.path.join(self.data_dir, 'train'), transform=train_transforms)
            self.val_dataset = self.BingRGBDataset(os.path.join(self.data_dir, 'val'), transform=val_transforms)
        
        if stage == 'test' or stage == 'predict' or stage is None:
            test_transforms = self.get_test_transforms()
            self.test_dataset = self.BingRGBDataset(os.path.join(self.data_dir, 'val'), transform=test_transforms)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_transforms(self):
        return A.Compose([
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])
    
    def get_test_transforms(self):
        return A.Compose([
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

    class BingRGBDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            
            self.image_files = [f for f in os.listdir(self.data_dir) if '_gt' not in f]
            
        def __len__(self):
            return len(self.image_files)
            
        def __getitem__(self, idx):
            image_file = self.image_files[idx]
            mask_file = self.image_files[idx].replace(".png", "_gt.png")
            
            image_path = os.path.join(self.data_dir, image_file)
            mask_path = os.path.join(self.data_dir, mask_file)
            
            image = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
            
            image[(mask == 0), :] = 0
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            return image, mask.long(), idx