import typing as T
import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms

from timm.models.mobilenetv3 import MobileNetV3
from timm.models.vision_transformer import VisionTransformer
from timm import create_model

from utils import count_parameters


def load_model_for_eval(model_name: str, num_classes: int, path: str) -> T.Union[MobileNetV3, VisionTransformer]:
    model: T.Union[MobileNetV3, VisionTransformer] = create_model(model_name, pretrained=False)
    if isinstance(model, MobileNetV3):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model.head = nn.Linear(model.head.in_features, num_classes)

    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def evaluate_model(model: T.Union[MobileNetV3, VisionTransformer], dataloader: data.DataLoader):
    model.eval()

    total = 0
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    inference_times = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            logits = model(images)
            end_time = time.time()

            inference_times.append(end_time - start_time)

            _, top5_preds = torch.topk(logits, k=5, dim=1)

            total += labels.size(0)
            correct_top1 += sum([labels[i] in top5_preds[i, :1] for i in range(labels.size(0))])
            correct_top3 += sum([labels[i] in top5_preds[i, :3] for i in range(labels.size(0))])
            correct_top5 += sum([labels[i] in top5_preds[i, :] for i in range(labels.size(0))])

    top1_accuracy = 100 * correct_top1 / total
    top3_accuracy = 100 * correct_top3 / total
    top5_accuracy = 100 * correct_top5 / total

    avg_inference_time_per_batch = sum(inference_times) / len(inference_times) * 1e3

    return top1_accuracy, top3_accuracy, top5_accuracy, avg_inference_time_per_batch


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 256
    val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Evaluate Models
    model = load_model_for_eval('mobilenetv3_small_100', num_classes=100, path='./student_models/mobilenet_trained_student_v3_final.pth').to(device)
    num_parameters_M = count_parameters(model) / 1e6
    print(f"Number of parameters => {num_parameters_M:.2f} M")
    
    top1_accuracy, top3_accuracy, top5_accuracy, avg_inference_time_per_batch = evaluate_model(model, val_loader)

    print(f"Top-1 Accuracy => {top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy => {top3_accuracy:.2f}%")
    print(f"Top-5 Accuracy => {top5_accuracy:.2f}%")
    print(f"Average Inference time per batch (batch size = {batch_size}) => {avg_inference_time_per_batch:.2f} ms")