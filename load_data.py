import torch
from torchvision import datasets, transforms


data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

def _get_train_data(data_dir):
    return datasets.ImageFolder(data_dir + '/train', transform=data_transforms)

def get_train_class_to_idx(data_dir):
    train_data = _get_train_data(data_dir)    
    return train_data.class_to_idx

def get_trainloader(data_dir):
    train_data = _get_train_data(data_dir)
    return torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
def get_validationloader(data_dir):
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms)
    return torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)

def get_testloader(data_dir):
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    return torch.utils.data.DataLoader(test_data, batch_size=32)
