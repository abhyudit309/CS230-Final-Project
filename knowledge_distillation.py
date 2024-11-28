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

from utils import count_parameters, prepare_student_for_training, load_teacher_for_distillation


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
    teacher_model: VisionTransformer,
    student_model: T.Union[MobileNetV3, VisionTransformer],
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    temperature: float,
    distillation_weight: float,
    val_freq: int,
    train_log_file: str = './student_logs/mobilenet_training_log_v1.txt',
    val_log_file: str = './student_logs/mobilenet_validation_log_v1.txt',
    save_path: str = './student_models/mobilenet_trained_student_v1.pth',
    final_path: str = './student_models/mobilenet_trained_student_v1_final.pth',
) -> None:
    assert 0.0 <= distillation_weight <= 1.0, 'Distillation weight should be in the range [0, 1]!'
    assert epochs % val_freq == 0, "Total epochs should be divisible by validation frequency!"

    teacher_model.eval()
    best_val_loss = np.inf
    optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(images)
            student_logits = student_model(images)

            p_teacher = F.softmax(teacher_logits / temperature, dim=1)
            p_student = F.log_softmax(student_logits / temperature, dim=1)
            distillation_loss = F.kl_div(p_student, p_teacher, reduction='batchmean') * (temperature ** 2)

            task_loss = F.cross_entropy(student_logits, labels)

            loss = distillation_weight * distillation_loss + (1.0 - distillation_weight) * task_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            pred = torch.argmax(student_logits, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}]: Training Loss => {avg_loss:.4f}, Accuracy => {accuracy:.2f}%\n')
        log_metrics(epoch, current_lr, avg_loss, accuracy, train_log_file)

        if (epoch + 1) % val_freq == 0:
            avg_val_loss, val_accuracy = validate_student(student_model, val_loader)
            print(f'Evaluation results for Epoch [{epoch + 1}/{epochs}]:')
            print(f'Validation Loss => {avg_val_loss:.4f}')
            print(f'Validation accuracy => {val_accuracy:.2f}%\n')
            log_metrics(epoch, current_lr, avg_val_loss, val_accuracy, val_log_file)

            # Checkpoint model
            if avg_val_loss < best_val_loss:
                print(f'Overwriting older model [val loss = {best_val_loss:.4f}] with model [val loss = {avg_val_loss:.4f}] at Epoch {epoch + 1}/{epochs}\n')
                best_val_loss = avg_val_loss
                torch.save(student_model.state_dict(), save_path)

        # Save final model as well
        if epoch + 1 == epochs:
            torch.save(student_model.state_dict(), final_path)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

    train_loader = data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Train Student Model
    teacher_model = load_teacher_for_distillation('vit_base_patch16_224', num_classes=100, path='./teacher_models/fine_tuned_teacher_v6_final.pth').to(device)
    student_model = prepare_student_for_training('mobilenetv3_small_100', num_classes=100).to(device)

    print(f'Number of parameters in teacher => {count_parameters(teacher_model) / 1e6}M')
    print(f'Number of parameters in student => {count_parameters(student_model) / 1e6}M')

    train_student(
        teacher_model,
        student_model,
        train_loader,
        val_loader, 
        learning_rate=1e-3, 
        weight_decay=1e-4, 
        epochs=50,
        temperature=3.0,
        distillation_weight=0.5,
        val_freq=1,
    )

    print('Training complete and models saved!')