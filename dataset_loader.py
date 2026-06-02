# dataset_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pose_extractor import extract_gesture_matrix_from_image
import config

class RobotGestureDataset(Dataset):
    def __init__(self, base_folder_path):
        self.samples = []
        # Get alphabetically sorted class names from directory folders
        self.class_names = sorted([d for d in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        print(f"Scanning directory: {base_folder_path}")
        print(f"Found classes mapped to IDs: {self.class_to_idx}")
        
        # Scan through folders to gather all image paths
        for cls_name in self.class_names:
            cls_folder = os.path.join(base_folder_path, cls_name)
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_folder, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))
                    
        print(f"Successfully cataloged {len(self.samples)} images total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Process the image into our custom 30x34 matrix format
        matrix = extract_gesture_matrix_from_image(img_path)
        
        # Add a placeholder 1-channel dimension to represent grayscale images: (1, 30, 34)
        matrix_tensor = torch.tensor(matrix).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return matrix_tensor, label_tensor

def get_data_loader(folder_path):
    dataset = RobotGestureDataset(folder_path)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)