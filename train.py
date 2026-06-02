# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RobotGestureResNet  # Crucial import

def train_and_personalize(mother_loader, patient_loader, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using computational hardware: {device}")
    
    model = RobotGestureResNet(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # -------------------------------------------------------------
    # PHASE 1: Train Mother Dataset (General Human Body Physics)
    # -------------------------------------------------------------
    print("Beginning Base 'Mother' Training Process...")
    
    for epoch in range(20): # Loop through your data 20 times
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for batch_idx, (matrices, labels) in enumerate(mother_loader):
            matrices, labels = matrices.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(matrices)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track training metrics
            running_loss += loss.item() * matrices.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_preds / total_samples) * 100
        print(f"Epoch [{epoch+1}/20] -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
            
    # Save the base generalized model
    torch.save(model.state_dict(), 'mother_model_weights.pth')
    print("--> Base 'Mother' model saved as 'mother_model_weights.pth'")
    
    # -------------------------------------------------------------
    # PHASE 2: Patient Adaptation (Personalizing for Weak/Rough Gestures)
    # -------------------------------------------------------------
    # Check if we actually have patient data loader elements to process
    if hasattr(patient_loader, 'dataset') and len(patient_loader.dataset) > 0:
        print("\nLoading Patient Profile. Personalizing deep network layers...")
        
        # Freeze initial layers so the model does not forget core human motion structures
        for param in model.init_conv.parameters(): param.requires_grad = False
        for param in model.layer1.parameters(): param.requires_grad = False
        
        # Fine-tuning optimizer
        personalize_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
        
        for epoch in range(5):
            model.train()
            for matrices, labels in patient_loader:
                matrices, labels = matrices.to(device), labels.to(device)
                
                personalize_optimizer.zero_grad()
                outputs = model(matrices)
                loss = criterion(outputs, labels)
                loss.backward()
                personalize_optimizer.step()
        print("--> Personalized Patient Deployment Matrix Completed successfully.")
    else:
        print("\n[Notice] Skipping personalization step. No patient datasets found.")
            
    return model