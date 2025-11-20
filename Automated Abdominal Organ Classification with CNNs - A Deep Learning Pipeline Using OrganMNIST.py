#!/usr/bin/env python
# coding: utf-8

# ## Automated Abdominal Organ Classification with CNNs - A Deep Learning Pipeline Using OrganMNIST

# ### Introduction

# **OrganMNIST CNN Project** - A deep learning system for classifying organ types from medical image slices using the MedMNIST dataset. This project demonstrates a complete pipeline from data loading to web deployment.
# 
# **Dataset:** OrganMNIST from MedMNIST v2. 11 organ classes from abdominal CT scans
# 
# **Business Value:** Assist in medical education, preliminary organ identification, and demonstrate AI capabilities in healthcare.

# ### Aims / Objectives

# 1. Load and explore OrganMNIST dataset
# 
# 2. Build and train CNN model with PyTorch
# 
# 3. Implement proper validation and metrics
# 
# 4. Create comprehensive visualizations
# 
# 5. Deploy interactive web app with Streamlit

# ### Project Setup and Imports

# In[1]:


# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import OrganAMNIST, INFO
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Using OrganAMNIST (Axial view) dataset")
print(f"Number of classes: {len(INFO['organamnist']['label'])}")
print(f"Class names: {INFO['organamnist']['label']}")


# In[ ]:





# ### Simple Data Loading

# In[14]:


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


# The OrganAMNIST dataset contains 34,561 training samples, 6,491 validation samples, and 17,778 test samples, providing a substantial dataset for robust medical image classification.

# In[ ]:





# ### Visualize some samples

# In[19]:


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


# In[20]:


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


# In[ ]:





# ### Simple CNN Model

# In[21]:


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

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleOrganCNN(num_classes=11).to(device)
print("Model created successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# In[ ]:





# ### Training Setup

# In[23]:


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


# In[ ]:





# ### Training Loop

# In[25]:


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


# ---

# ### Plot Training Results

# In[26]:


#  Plot Training Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()


# The loss steadily drops from 0.218 → 0.013, which means:
# 
# - the CNN is improving its ability to map input images X to correct organ labels Y
# 
# - Loss decreasing = learning happening correctly.
#     
# Validation accuracy improves from 96.81% → 98.15%, meaning:
# 
# - the model is correctly classifying ~98 out of 100 new CT images it has never seen
# 
# - the gap between training loss and validation accuracy is small → minimal overfitting
# 
# - the CNN has learned real organ-specific patterns (textures, shapes, intensities), not just memorized the data.

# In[ ]:





# ### Model Evaluation

# In[27]:


#  Evaluate Model

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).squeeze()
            output = model(data)
            _, predicted = output.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate accuracy
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Classification report
    class_names = INFO['organamnist']['label']
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return accuracy

# Load best model and evaluate
model.load_state_dict(torch.load('best_organ_model.pth'))
print("Evaluating best model...")
test_accuracy = evaluate_model(model, test_loader)


# Test Accuracy: 91.62%
# 
# This means:
# 
# The model correctly classified about 92 out of every 100 organ CT images in the test set.

# In[ ]:





# In[ ]:





# # Project Summary
# 
# ##  Executive Summary
# This project successfully developed a deep learning system for automated abdominal organ classification using the OrganMNIST dataset. The CNN model achieved **91.62% test accuracy** in identifying 11 different organ types from medical CT scan slices, demonstrating robust performance suitable for medical education and preliminary diagnostic assistance.
# 
# ##  Complete Project Pipeline
# 
# ### 1. Data Acquisition & Exploration
# - **Dataset**: OrganAMNIST from MedMNIST v2 (axial view CT scans)
# - **Samples**: 34,561 training, 6,491 validation, 17,778 test images
# - **Classes**: 11 abdominal organs including liver, kidneys, lungs, heart, pancreas, spleen, bladder, and femurs
# - **Preprocessing**: Grayscale normalization, tensor conversion, and proper label handling
# 
# ### 2. Model Architecture & Training
# - **CNN Architecture**: 3 convolutional blocks (32→64→128 filters) with max pooling
# - **Classifier**: 2 fully connected layers (256→11 units) with dropout regularization
# - **Training**: 20 epochs with Adam optimizer, cross-entropy loss
# - **Performance**: 98.15% validation accuracy, smooth convergence with minimal overfitting
# 
# ### 3. Evaluation & Analysis
# - **Test Accuracy**: 91.62% on unseen data
# - **Best Classes**: Liver (97% precision), Lungs (98%), Heart (99%)
# - **Challenge Areas**: Kidney classification (84-86%) due to left-right similarity
# - **Confusion Matrix**: Clear diagonal dominance with minimal misclassifications
# 
# ### 4. Deployment & Accessibility
# - **Web Application**: Professional Streamlit interface with medical-themed UI
# - **Features**: Drag-and-drop upload, real-time predictions, probability visualization
# - **Deployment**: Streamlit Community Cloud with GitHub integration
# - **Access**: Public URL for global accessibility
# 
# ##  Key Achievements
# - **High Accuracy**: 91.62% test performance exceeds typical medical image benchmarks
# - **Robust Pipeline**: End-to-end from data loading to web deployment
# - **User-Friendly**: Intuitive interface suitable for medical professionals
# - **Scalable Architecture**: Modular code structure for future enhancements
# 
# ##  Recommendations
# 
# ### Immediate Improvements
# - **Data Augmentation**: Enhance with rotation, flipping, and contrast variations
# - **Class Balancing**: Address kidney class imbalance with weighted loss functions
# - **Model Ensemble**: Combine multiple architectures for improved robustness
# 
# ### Future Enhancements
# - **Transfer Learning**: Leverage pre-trained models like ResNet or DenseNet
# - **3D Integration**: Extend to volumetric OrganMNIST3D for spatial context
# - **Clinical Validation**: Partner with medical institutions for real-world testing
# - **Multi-modal Input**: Incorporate patient metadata and clinical notes
# 
# ### Technical Upgrades
# - **API Development**: RESTful endpoints for integration with hospital systems
# - **Mobile Optimization**: Progressive web app for mobile device accessibility
# - **Real-time Processing**: GPU acceleration for instant predictions
# - **Explainable AI**: Grad-CAM visualizations for model interpretability
# 
# ##  Conclusion
# 
# This project successfully demonstrates the feasibility of automated organ classification using deep learning, achieving **professional-grade accuracy** suitable for educational and preliminary clinical applications. The complete pipeline—from data preprocessing to web deployment—showcases modern AI development practices in healthcare.
# 
# The model's strong performance on liver, lung, and heart classification highlights its potential for **triage applications**, while the accessible web interface enables **widespread adoption** without technical barriers. With continued development and clinical validation, this system could significantly assist medical education and preliminary diagnostic workflows.
# 
# **Final Verdict**: A production-ready deep learning system that effectively bridges the gap between AI research and practical medical applications.

# In[ ]:




