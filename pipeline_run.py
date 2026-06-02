# pipeline_run.py
import torch
import config
from dataset_loader import get_data_loader
from model import RobotGestureResNet
from train import train_and_personalize

def main():
    print("=============================================")
    print("   Starting Robotic Assistant AI Pipeline   ")
    print("=============================================\n")
    
    # 1. Load Datasets
    print("[Step 1/3] Loading your custom image datasets...")
    mother_loader = get_data_loader(config.DATASET_PATH_MOTHER)
    patient_loader = get_data_loader(config.DATASET_PATH_PATIENT)
    
    if mother_loader is None:
        print("\n[CRITICAL ERROR] Mother dataset folder is missing or contains no images.")
        print(f"Please check your path configuration inside config.py: {config.DATASET_PATH_MOTHER}")
        return
        
    # 2. Initialize the Network Architecture
    print("\n[Step 2/3] Constructing Spatio-Temporal Residual Network Architecture...")
    # Update config automatically if classes found don't match default placeholder
    num_classes = len(mother_loader.dataset.class_names)
    print(f"Configuring output layer node matrix for {num_classes} unique gestures.")
    
    # 3. Trigger Training Pipeline
    print("\n[Step 3/3] Compiling training execution engine...")
    # If a patient dataset doesn't exist yet, pass a dummy list to prevent breaking
    if patient_loader is None:
        print("[Notice] No custom patient profile data found. Training base Mother model only.")
        patient_loader = []
        
    trained_model = train_and_personalize(mother_loader, patient_loader, num_classes=num_classes)
    
    print("\n=============================================")
    print("   Pipeline Execution Finished Successfully!  ")
    print("=============================================")

if __name__ == '__main__':
    main()