# evaluate_accuracy.py
import torch
import numpy as np
from dataset_loader import get_data_loader
from model import RobotGestureResNet
import config

def calculate_total_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating system on: {device}")

    # 1. Load the dataset
    mother_loader = get_data_loader(config.DATASET_PATH_MOTHER)
    if mother_loader is None:
        print("Error: Mother dataset not found.")
        return

    class_names = mother_loader.dataset.class_names
    num_classes = len(class_names)

    # 2. Reconstruct model and load your saved weights
    model = RobotGestureResNet(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load('mother_model_weights.pth', map_location=device))
    except FileNotFoundError:
        print("Error: 'mother_model_weights.pth' not found. Please run pipeline_run.py first.")
        return
        
    model.eval()

    total_correct = 0
    total_samples = 0
    
    # Track matrix to count matches per gesture class
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    print("\nCrunching numbers across all cataloged data samples...")
    
    with torch.no_grad():
        for matrices, labels in mother_loader:
            matrices, labels = matrices.to(device), labels.to(device)
            outputs = model(matrices)
            _, predicted = torch.max(outputs, 1)
            
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            # Calculate metrics for individual classes
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    # 3. Calculate Final Total Scores
    overall_accuracy = (total_correct / total_samples) * 100
    
    print("\n=============================================")
    print("      TOTAL SYSTEM ACCURACY REPORT           ")
    print("=============================================")
    print(f"🔥 OVERALL TOTAL ACCURACY: {overall_accuracy:.2f}%")
    print(f"Total Images Processed: {total_samples}")
    print("---------------------------------------------")
    print("Accuracy Breakdown by Gesture Class:")
    print("---------------------------------------------")
    
    for i in range(num_classes):
        if class_total[i] > 0:
            cls_acc = (class_correct[i] / class_total[i]) * 100
            print(f" * {class_names[i]:<15} : {cls_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f" * {class_names[i]:<15} : No samples found")
    print("=============================================\n")

if __name__ == '__main__':
    calculate_total_accuracy()