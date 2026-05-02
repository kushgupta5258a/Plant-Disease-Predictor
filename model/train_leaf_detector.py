import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "data/dataset/leaf vs non leaf"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# TRANSFORMS (Augmentation)
# -----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# LOAD DATASET
# -----------------------
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

class_names = full_dataset.classes  # ['leaf', 'non_leaf'] or similar
num_classes = len(class_names)

print(f"Total classes: {num_classes}")
print(f"Classes: {class_names}")
print(f"Total images: {len(full_dataset)}\n")

# Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply val transform separately
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# MODEL (EfficientNet Binary Classifier)
# -----------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier for binary classification (2 classes: leaf, non_leaf)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(DEVICE)

# -----------------------
# LOSS & OPTIMIZER
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

# -----------------------
# TRAINING LOOP
# -----------------------
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}, Val Acc: {acc:.4f}")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model/leaf_detector.pkl")

print("\nTraining complete!")
print("Best Accuracy:", best_acc)
print("Model saved as: model/leaf_detector.pkl")
