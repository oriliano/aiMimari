import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    IMG_SIZE = 224  

    # augmentation + preprocess
    train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)
    ], p=0.7),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5)
    ], p=0.6),
    transforms.RandomPerspective(distortion_scale=0.05, p=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
    ], p=0.15),
    transforms.ToTensor(),
    # Gürültü sadece tensor aşamasında eklenir!
    transforms.RandomApply([
        transforms.Lambda(lambda img: img + 0.01 * torch.randn_like(img))
    ], p=0.10),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

    # validation için sadece preprocess
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # test için yalnızca preprocess
    test_transforms = val_transforms

    data_dir = "splitted_dataset"

    train_dataset = ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
    val_dataset   = ImageFolder(root=f"{data_dir}/val",   transform=val_transforms)
    test_dataset  = ImageFolder(root=f"{data_dir}/test",  transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sınıf sayısını otomatik algıla
    num_classes = len(train_dataset.classes)
    print(f"Sınıf sayısı: {num_classes}")
    print("Sınıflar:", train_dataset.classes)

    model = models.resnet18(pretrained=True)
    # Son fc katmanına Dropout ekleyelim (overfitting azaltır)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Learning rate scheduler ekleyelim (val loss stagnate ederse)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    num_epochs = 20
    best_val_acc = 0.0
    patience = 7
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step(val_loss)  # LR scheduler ile val_loss takip

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "best_architecture.pt")
            best_val_acc = val_acc
            patience_counter = 0
            print("Yeni en iyi model kaydedildi!")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping tetiklendi.")
            break

    print("Eğitim tamamlandı.")

if __name__ == "__main__":
    main()
###