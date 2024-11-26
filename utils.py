import torch.nn as nn
from timm import create_model
from timm.models.vision_transformer import VisionTransformer


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def initialize_new_head(layer: nn.Linear):
    assert isinstance(layer, nn.Linear)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


def prepare_model_for_finetune(model_name: str, num_classes: int) -> VisionTransformer:
    teacher_model: VisionTransformer = create_model(model_name, pretrained=True)
    teacher_model.head = nn.Linear(teacher_model.head.in_features, num_classes)
    teacher_model.head.apply(initialize_new_head)

    for param_name, param in teacher_model.named_parameters():
        if 'head' not in param_name:
            param.requires_grad = False

    return teacher_model