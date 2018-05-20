import torch
from torchvision import models
from torch import nn
from collections import OrderedDict


def get_model(arch, hidden_units):
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = _create_classifier(hidden_units)

    return model
    

def _create_classifier(hidden_units):
    classifier_layers = []

    linear_layers_sizes = zip(hidden_units[:-1], hidden_units[1:])
    counter = 1
    for h1, h2 in linear_layers_sizes:
        if(len(classifier_layers) > 0):
            classifier_layers.extend([ ('relu' + str(counter), nn.ReLU()) ])
            counter += 1

        classifier_layers.extend([ ('fc' + str(counter), nn.Linear(h1, h2)) ])

    classifier_layers.extend([ ('output', nn.LogSoftmax(dim=1)) ])

    return nn.Sequential(OrderedDict(classifier_layers))


def save_trained_model(model, arch, class_to_idx, hidden_units, epochs, learning_rate, filename):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate
    }
    torch.save(checkpoint, filename)


def load_trained_model(filename):
    checkpoint = torch.load(filename)
    model = get_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
