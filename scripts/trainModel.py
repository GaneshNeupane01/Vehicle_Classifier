import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from roboflow import Roboflow


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

class VehicleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        for idx, cls in enumerate(self.class_names):
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                self.images.append(os.path.join(cls_dir, img))
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

#Load Roboflow Dataset
rf = Roboflow(api_key="YOUR_API_KEY_HERE")
project = rf.workspace("paul-guerrie-tang1").project("vehicle-classification-eapcd")
dataset = project.version(19).download("folder")

train_dir = os.path.join(dataset.location, "train")
val_dir   = os.path.join(dataset.location, "valid")


# HEAD STAGE: simpler augmentation
head_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# FINE-TUNING STAGE: harder augmentation + larger input
ft_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


#Load Dataset
train_dataset = VehicleDataset(train_dir, head_transform)
val_dataset   = VehicleDataset(val_dir, head_transform)
num_classes = len(train_dataset.class_names)
print("Detected classes:", train_dataset.class_names)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)


#Class Weights
counts = Counter(train_dataset.labels)
total = sum(counts.values())
class_weights = torch.tensor([total / counts[i] for i in range(num_classes)], dtype=torch.float).to(device)

model = timm.create_model("convnext_tiny", pretrained=True, num_classes=num_classes).to(device)

#Stage 1: Head Only Training
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scaler = torch.cuda.amp.GradScaler()

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            pred = torch.argmax(outputs, 1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="weighted")
    rec = recall_score(targets, preds, average="weighted")
    f1 = f1_score(targets, preds, average="weighted")
    return acc, prec, rec, f1

best_acc = 0
best_model_path = MODEL_DIR / "best_vehicle_model.pth"

print("\n=== Stage 1: Head Training ===")
for epoch in range(8):  # longer for better adaptation
    print(f"\n[Head] Epoch {epoch+1}/8")
    loss = train_one_epoch(model, train_loader)
    acc, prec, rec, f1 = validate(model, val_loader)
    scheduler.step()
    print(f"Loss: {loss:.4f} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save({
            "model": model.state_dict(),
            "classes": train_dataset.class_names
        }, str(best_model_path))

#Stage 2: Fine-Tuning Last ConvNeXt Stage
# Use harder transforms for FT stage
train_dataset.transform = ft_transform
val_dataset.transform = ft_transform

for name, param in model.named_parameters():
    if "stages.3" in name:
        param.requires_grad = True

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

print("\n=== Stage 2: Fine-Tuning Last Stage ===")
for epoch in range(12):
    print(f"\n[Fine-Tuning] Epoch {epoch+1}/12")
    loss = train_one_epoch(model, train_loader)
    acc, prec, rec, f1 = validate(model, val_loader)
    scheduler.step()
    print(f"Loss: {loss:.4f} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save({
            "model": model.state_dict(),
            "classes": train_dataset.class_names
        }, str(best_model_path))

#Stage 3: Unfreeze Stage 2 for Full Fine-Tuning
for name, param in model.named_parameters():
    if "stages.2" in name:
        param.requires_grad = True

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

print("\n=== Stage 3: Optional Full Fine-Tuning Stage 2 ===")
for epoch in range(8):
    print(f"\n[Full Fine-Tune] Epoch {epoch+1}/8")
    loss = train_one_epoch(model, train_loader)
    acc, prec, rec, f1 = validate(model, val_loader)
    scheduler.step()
    print(f"Loss: {loss:.4f} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save({
            "model": model.state_dict(),
            "classes": train_dataset.class_names
        }, str(best_model_path))

def predict(image_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = ft_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(img), 1).item()
    return train_dataset.class_names[pred]
