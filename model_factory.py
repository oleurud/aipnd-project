import torch
from torchvision import models
from torch import nn
from collections import OrderedDict


def get_model(arch):
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # change classifier
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1920, 1000)),
                                ('relu1', nn.ReLU()),
                                ('fc2', nn.Linear(1000, 500)),
                                ('relu2', nn.ReLU()),
                                ('fc3', nn.Linear(500, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    model.classifier = classifier
    
    model.arch = arch

    return model

def save_trained_model(model, class_to_idx, filename):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'arch': model.arch
    }
    torch.save(checkpoint, filename)

def load_trained_model(filename):
    checkpoint = torch.load()
    model = get_model(checkpoint.arch)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
