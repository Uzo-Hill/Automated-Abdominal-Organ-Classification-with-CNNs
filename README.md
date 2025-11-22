# Automated-Abdominal-Organ-Classification-with-CNNs

A deep learning pipeline for classifying 11 abdominal organs from CT scan slices using PyTorch and Streamlit deployment.
---

![Sample Images](https://github.com/Uzo-Hill/Automated-Abdominal-Organ-Classification-with-CNNs/blob/main/project%20image/Cover_Image.jpg)
---
## Project Overview

This project builds an end-to-end Convolutional Neural Network (CNN) for classifying abdominal organs from the OrganMNIST dataset (MedMNIST v2).
It covers the full machine-learning workflow:

- Dataset loading & preprocessing

- Sample visualization

- Custom CNN architecture (PyTorch)

- Model training & validation

- Performance evaluation (classification report & confusion matrix)

- Deployment-ready pipeline

The model achieves ~98% validation accuracy and 91.62% test accuracy.

---
## Dataset
Source: MedMNIST v2 — OrganAMNIST (Axial CT slices)
Classes: 11 abdominal organs
Samples:

- Train: 34,561

- Validation: 6,491

- Test: 17,778

Example Classes: liver, kidney-left, kidney-right, lung-left, lung-right, heart, spleen, femur-left/right, bladder, pancreas.



---
## Data Loading & Preprocessing

Data was loaded using the OrganAMNIST API and transformed using:

- transforms.ToTensor()

- transforms.Normalize(mean=[0.5], std=[0.5])

This ensures uniform pixel scaling and optimal training behavior.


```python
# Simple Data Loading

def load_data():
    """Load the OrganAMNIST dataset"""
    # Simple transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load datasets
    train_dataset = OrganAMNIST(split='train', transform=transform, download=True)
    val_dataset = OrganAMNIST(split='val', transform=transform, download=True)
    test_dataset = OrganAMNIST(split='test', transform=transform, download=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = load_data()
```

## Visualizing Sample Images

Random samples were plotted to confirm:

- Correct data loading

- Proper label mapping

- Pixel normalization

```python
# Debug: Check the labels
print("Debugging labels...")
for i in range(5):
    _, label = train_dataset[i]
    print(f"Sample {i}: label type={type(label)}, label value={label}")

# Check class names
class_names = INFO['organamnist']['label']
print(f"\nClass names: {class_names}")
print(f"Number of classes: {len(class_names)}")
print(f"Class names indices: {list(range(len(class_names)))}")


# Visualize some samples 
def visualize_samples(dataset, title="Dataset Samples"):
    """Visualize sample images"""
    class_names = INFO['organamnist']['label']
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(10):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        
        # Convert tensor to numpy for display
        img_np = image.numpy().squeeze()
        
        # Convert label to integer and adjust for dictionary keys
        if isinstance(label, np.ndarray):
            label = label.item()  # Get the scalar value from the array
        
        # Labels are 1-11, but class_names uses string keys '0'-'10'
        # So we need to convert: label 1 -> '0', label 2 -> '1', etc.
        label_key = str(label - 1)
        
        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(f'{class_names[label_key]}')
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_samples(train_dataset, "OrganAMNIST Training Samples")
```

![Sample Images](https://github.com/Uzo-Hill/Automated-Abdominal-Organ-Classification-with-CNNs/blob/main/project%20image/Image_Samples.PNG)

---
## CNN Model Architecture
A simple yet powerful CNN was built with:

Convolutional Layers

- 3 blocks: 32 → 64 → 128 filters

- Each block: Conv2d → ReLU → MaxPool2d

- Spatial downsampling by pooling

Classifier Layers

- Flatten

- Linear(128*3*3 → 256)

- ReLU + Dropout(0.5)

- Linear(256 → 11 classes)

```python
# Simple CNN Model

class SimpleOrganCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SimpleOrganCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second block  
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x
```

---

## Training & Optimization

Training setup:

- Loss Function: CrossEntropyLoss

-Optimizer: Adam(lr=0.001)

- Epochs: 20

- Batch size: 64

Training tracked:

- Loss per epoch
- Validation accuracy
- Best-model checkpointing

```python
# Training Setup

def setup_training():
    """Setup data loaders and optimizer"""
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    return train_loader, val_loader, test_loader, criterion, optimizer

train_loader, val_loader, test_loader, criterion, optimizer = setup_training()
```
---
```python
#  Simple Training Loop 

def train_simple_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    """Simple training function"""
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # SIMPLE FIX: Convert target to long and flatten
            target = target.long().view(-1)  # This handles all cases
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target = target.long().view(-1)  # Same fix for validation
                    
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        train_loss_avg = train_loss / len(train_loader)
        
        train_losses.append(train_loss_avg)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Loss: {train_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc == max(val_accuracies):
            torch.save(model.state_dict(), 'best_organ_model.pth')
    
    return train_losses, val_accuracies

print("Starting training...")
train_losses, val_accuracies = train_simple_model(
    model, train_loader, val_loader, criterion, optimizer, epochs=20
)
```
![Sample Images](https://github.com/Uzo-Hill/Automated-Abdominal-Organ-Classification-with-CNNs/blob/main/project%20image/Model_Training.PNG)


---
## Training Performance

The model converged smoothly:

- Loss: 0.6375 → 0.0135

- Validation Accuracy: 95.59% → 98.41%

- Minimal overfitting

- Stable gradient flow


![Training Loss&vaidation accuracy](https://github.com/Uzo-Hill/Automated-Abdominal-Organ-Classification-with-CNNs/blob/main/project%20image/Loss_Validation_Accuracy.PNG)
---

## Model Evaluation

The best model was loaded and tested on unseen data.

Test Accuracy: 90.12%

![Training Loss&vaidation accuracy](https://github.com/Uzo-Hill/Automated-Abdominal-Organ-Classification-with-CNNs/blob/main/project%20image/Mode_Accuracy_Evaluation.PNG)


![Training Loss&vaidation accuracy](https://github.com/Uzo-Hill/Automated-Abdominal-Organ-Classification-with-CNNs/blob/main/project%20image/Confusion_Matrix.PNG)

---

## Key Insights:

- Strong performance on liver, lungs, and heart (97–99% precision)

- Lower performance on left/right kidney due to structural similarity

- Confusion matrix shows dominant diagonal (strong predictions)

---
## Web Deployment
Streamlit Application Features:

- Drag-and-drop image upload

- Real-time predictions

- Probability distribution visualization

- Medical-themed professional UI

- Mobile-responsive design
  

Deployment: Streamlit Community Cloud with GitHub integration

- Public URL: Accessible globally

- Auto-deployment: Updates with Git pushes

- Free Tier: No hosting costs

---

## Future Enhancements
- Integration: REST API / mobile app deployment
- Grad-CAM for model interpretability
---

## Conclusion

This project demonstrates a full, professional deep-learning workflow for medical image classification, achieving:

- High accuracy (90.12%)
- Robust CNN architecture
- Strong generalization
- Scalable deployment pipeline

It represents a solid, production-ready baseline for medical image AI applications.

---
⭐ Star this repo if you found it helpful!










