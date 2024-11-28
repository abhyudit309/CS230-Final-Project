import typing as T

import torch
import torch.nn as nn
from timm import create_model
from timm.models.mobilenetv3 import MobileNetV3
from timm.models.vision_transformer import VisionTransformer


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def initialize_new_head(layer: nn.Linear):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


def prepare_teacher_for_finetune(model_name: str, num_classes: int, use_dropout: bool = False) -> VisionTransformer:
    teacher_model: VisionTransformer = create_model(model_name, pretrained=True)
    if use_dropout:
        teacher_model.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(teacher_model.head.in_features, num_classes)
        )
    else:
        teacher_model.head = nn.Linear(teacher_model.head.in_features, num_classes)
    teacher_model.head.apply(initialize_new_head)

    for param_name, param in teacher_model.named_parameters():
        if 'head' not in param_name:
            param.requires_grad = False

    return teacher_model


def prepare_student_for_training(model_name: str, num_classes: int) -> T.Union[MobileNetV3, VisionTransformer]:
    student_model: T.Union[MobileNetV3, VisionTransformer] = create_model(model_name, pretrained=False)
    if isinstance(student_model, MobileNetV3):
        student_model.classifier = nn.Linear(student_model.classifier.in_features, num_classes)
        student_model.classifier.apply(initialize_new_head)
    else:
        student_model.head = nn.Linear(student_model.head.in_features, num_classes)
        student_model.head.apply(initialize_new_head)

    return student_model


def load_teacher_for_distillation(model_name: str, num_classes: int, path: str) -> VisionTransformer:
    teacher_model: VisionTransformer = create_model(model_name, pretrained=False)
    teacher_model.head = nn.Linear(teacher_model.head.in_features, num_classes)

    teacher_model.load_state_dict(torch.load(path, weights_only=True))
    return teacher_model