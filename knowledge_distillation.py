import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
from torchvision import datasets, transforms
from timm import create_model

def count_parameters(model: nn.Module):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

teacher_model = create_model('vit_base_patch16_224', pretrained=True)
teacher_model.eval()

student_model_mobilenet = create_model('mobilenetv3_small_100', pretrained=False, num_classes=1000)

student_model_vit = create_model('deit_tiny_patch16_224', pretrained=False, num_classes=1000)

print(type(teacher_model), type(student_model_mobilenet), type(student_model_vit))
print(f"Num of param in teacher => {count_parameters(teacher_model) / 1e6}M")
print(f"Num of param in student mobilenet => {count_parameters(student_model_mobilenet) / 1e6}M")
print(f"Num of param in student vit => {count_parameters(student_model_vit) / 1e6}M")

def distillation_loss(logits_student, logits_teacher, temperature=3.0):
    p_teacher = nn.functional.softmax(logits_teacher / temperature, dim=1)
    p_student = nn.functional.log_softmax(logits_student / temperature, dim=1)
    loss = nn.functional.kl_div(p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
    return loss

def feature_distillation_loss(student_features, teacher_features):
    loss = nn.functional.mse_loss(student_features, teacher_features)
    return loss

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# training
student_model = student_model_mobilenet
student_model = student_model.to('cpu')
teacher_model = teacher_model.to('cpu')

optimizer = optim.Adam(student_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_distillation(teacher: nn.Module, student: nn.Module, train_loader, optimizer, criterion, temperature=3.0, epochs=10):
    teacher.eval()
    student.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to('cpu'), labels.to('cpu')

            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)

            loss_distillation = distillation_loss(student_logits, teacher_logits, temperature)
            loss_task = nn.functional.cross_entropy(student_logits, labels)
            print("1", teacher_logits.shape)
            print("2", student_logits.shape)
            print("3", labels.shape)
            loss = loss_distillation + loss_task

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

train_distillation(teacher_model, student_model, train_loader, optimizer, criterion)   