import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class TryOnDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        self.cloth_dir = os.path.join(data_dir, 'cloth')
        self.image_mask_dir = os.path.join(data_dir, 'image-mask')
        self.cloth_mask_dir = os.path.join(data_dir, 'cloth-mask')
        self.pose_dir = os.path.join(data_dir, 'pose')
        
        self.image_list = os.listdir(self.image_dir)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        cloth_path = os.path.join(self.cloth_dir, img_name)
        image_mask_path = os.path.join(self.image_mask_dir, img_name.replace('.jpg', '.png'))
        cloth_mask_path = os.path.join(self.cloth_mask_dir, img_name.replace('.jpg', '.png'))
        pose_path = os.path.join(self.pose_dir, img_name.replace('.jpg', '_keypoints.json'))
        
        image = Image.open(img_path).convert('RGB')
        cloth = Image.open(cloth_path).convert('RGB')
        image_mask = Image.open(image_mask_path).convert('L')
        cloth_mask = Image.open(cloth_mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            cloth = self.transform(cloth)
            image_mask = self.transform(image_mask)
            cloth_mask = self.transform(cloth_mask)
        
        # Load pose data (you may need to implement this based on your pose format)
        pose = torch.zeros(17, 2)  # Placeholder, replace with actual pose data
        
        return image, cloth, image_mask, cloth_mask, pose

def get_dataloader(data_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = TryOnDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return dataloader
