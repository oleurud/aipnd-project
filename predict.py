#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
import json
import argparse
import model_factory
from torch.autograd import Variable


def main():
    image_path, checkpoint, top_k, category_names, gpu = get_input_args()

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = model_factory.load_trained_model(checkpoint)
    idx_to_class = {i:k for k, i in model.class_to_idx.items()}

    probs, classes = predict(image_path, model, idx_to_class, top_k, gpu)

    for prob, classe in zip(probs, classes):
        print(cat_to_name[classe], prob)


def get_input_args():
    parser = argparse.ArgumentParser(description='description')

    parser.add_argument('image_path',
                        metavar='image_path',
                        type=str,
                        help='Image path')

    parser.add_argument('checkpoint',
                        metavar='checkpoint',
                        default='./cli_checkpoint.pth',
                        type=str,
                        help='checkpoint')

    parser.add_argument('--top_k',
                        type=int,
                        default=5,
                        help='Top k most likely classes')
    
    parser.add_argument('--category_names',
                        type=str,
                        default='./cat_to_name.json',
                        help='Mapping of categories to real names')

    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use gpu')

    
    args = parser.parse_args()

    if args.gpu == True and torch.cuda.is_available() == False:
        args.gpu = False
        print("GPU is not avaliable")
    
    if args.gpu == False and torch.cuda.is_available() == True:
        print("GPU avaliable. You should use it")

    return args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu


def predict(image_path, model, idx_to_class, topk, cuda):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    

    image = Image.open(image_path)
    image = process_image(image)
    image = torch.FloatTensor([image])
    
    if cuda:
        model.cuda()
        image = image.cuda()
    
    model.eval()

    output = model.forward(Variable(image))
    
    # top predictions
    if cuda:
        all_probs = torch.exp(output).data.cpu().numpy()[0]
    else:
        all_probs = torch.exp(output).data.numpy()[0]
    topk_index = np.argsort(all_probs)[-topk:][::-1] 
    topk_class = [idx_to_class[x] for x in topk_index]
    topk_prob = all_probs[topk_index]
    
    return topk_prob, topk_class


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # resize shortest side to 256
    width, height = image.size
    if height > width:
        new_width = 256
        new_height = int(np.floor(256 * height / width))
    elif height < width:
        new_width = int(np.floor(256 * height / width))
        new_height = 256
    image = image.resize((new_width, new_height))
    
    # crop out the center 224x224 
    left = (new_width - 224)/2
    top = (new_height - 224)/2
    right = (new_width + 224)/2
    bottom = (new_height + 224)/2
    image = image.crop((left, top, right, bottom))
    
    # normalize
    image = np.array(image)
    image = image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # reorder layers
    image = image.transpose(2, 0, 1)
    
    return image



# Call to main function to run the program
if __name__ == "__main__":
    main()
