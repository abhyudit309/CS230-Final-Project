import typing as T
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from tqdm import tqdm
from timm.models.mobilenetv3 import MobileNetV3
from timm.models.vision_transformer import VisionTransformer

from utils import prepare_student_for_training


def log_metrics(epoch: int, lr: float, loss: float, accuracy: float, file_path: str) -> None:
    with open(file_path, 'a') as file:
        file.write(f'Epoch: {epoch + 1}, LR => {lr:.6f}, Loss => {loss:.4f}, Accuracy => {accuracy:.2f}%\n')


def validate_student(model: T.Union[MobileNetV3, VisionTransformer], val_loader: data.DataLoader) -> T.Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            val_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return avg_val_loss, val_accuracy


def train_student(
    model: T.Union[MobileNetV3, VisionTransformer],
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    val_freq: int,
    train_log_file: str = './student_logs/mobilenet_training_log_v5.txt',
    val_log_file: str = './student_logs/mobilenet_validation_log_v5.txt',
    save_path: str = './student_models/mobilenet_trained_student_v5.pth',
    final_path: str = './student_models/mobilenet_trained_student_v5_final.pth',
) -> None:
    assert epochs % val_freq == 0, "Total epochs should be divisible by validation frequency!"

    best_val_loss = np.inf
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            pred = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}]: Training Loss => {avg_loss:.4f}, Accuracy => {accuracy:.2f}%\n')
        log_metrics(epoch, current_lr, avg_loss, accuracy, train_log_file)

        if (epoch + 1) % val_freq == 0:
            avg_val_loss, val_accuracy = validate_student(model, val_loader)
            print(f'Evaluation results for Epoch [{epoch + 1}/{epochs}]:')
            print(f'Validation Loss => {avg_val_loss:.4f}')
            print(f'Validation accuracy => {val_accuracy:.2f}%\n')
            log_metrics(epoch, current_lr, avg_val_loss, val_accuracy, val_log_file)

            # Checkpoint model
            if avg_val_loss < best_val_loss:
                print(f'Overwriting older model [val loss = {best_val_loss:.4f}] with model [val loss = {avg_val_loss:.4f}] at Epoch {epoch + 1}/{epochs}\n')
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)

        # Save final model as well
        if epoch + 1 == epochs:
            torch.save(model.state_dict(), final_path)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading dataset
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=train_transforms, download=True)
    val_dataset = datasets.CIFAR100(root='./data', train=False, transform=val_transforms, download=True)

    train_loader = data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Train Student Model
    student_model = prepare_student_for_training('mobilenetv3_small_100', num_classes=100).to(device)
    train_student(student_model, train_loader, val_loader, learning_rate=1e-3, weight_decay=1e-4, epochs=50, val_freq=1)
    print('Training complete and models saved!')