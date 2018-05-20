#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import load_data
import model_factory
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable



def main():
    data_dir, save_dir, arch, learning_rate, epochs, gpu = get_input_args()
    
    trainloader = load_data.get_trainloader(data_dir)
    validationloader = load_data.get_validationloader(data_dir)

    model = model_factory.get_model(arch)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    train(model, trainloader, validationloader, optimizer, criterion, epochs, gpu)

    model_factory.save_trained_model(model, load_data.get_train_class_to_idx(data_dir), save_dir)


def get_input_args():
    parser = argparse.ArgumentParser(description='description')

    parser.add_argument('data_dir',
                        metavar='data_dir',
                        type=str,
                        default='./flowers/',
                        help='Path to save checkpoints')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./cli_checkpoint.pth',
                        help='Path to save checkpoints')

    parser.add_argument('--arch',
                        type=str,
                        default='densenet201',
                        help='Architecture')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Learning rate')

    # it has no sense since we are using models from torch
    # parser.add_argument('--hidden_units',
    #                     type=int,
    #                     default=3,
    #                     help='Hidden units')

    parser.add_argument('--epochs',
                        type=int,
                        default=3,
                        help='Epochs')

    parser.add_argument('--gpu',
                        type=bool,
                        default=False,
                        help='Use gpu')
                    

    args = parser.parse_args()

    if args.gpu == True and torch.cuda.is_available() == False:
        args.gpu = False
        print("GPU is not avaliable")
    
    if args.gpu == False and torch.cuda.is_available() == True:
        print("GPU avaliable. You should use it")

    return args.data_dir, args.save_dir, args.arch, args.learning_rate, args.epochs, args.gpu


def train(model, trainloader, validationloader, optimizer, criterion, epochs, cuda = False):
    steps = 0 
    running_loss = 0 
    print_every = 1
    
    if cuda:
        model.cuda()

    for e in range(epochs):

        # Model in training mode, dropout is on
        model.train()        

        for ii, (inputs, labels) in enumerate(trainloader):
            inputs, labels = Variable(inputs), Variable(labels)
            steps += 1

            optimizer.zero_grad()

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss, accuracy = validation(model, validationloader, criterion, cuda)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(validationloader)),
                    "Test Accuracy: {:.3f}".format(accuracy/len(validationloader)))

                running_loss = 0


def validation(model, validationloader, criterion, cuda):
    # Model in inference mode, dropout is off
    model.eval()

    accuracy = 0
    test_loss = 0
    for ii, (inputs, labels) in enumerate(validationloader):

        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)


        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output).data
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    
    # Make sure dropout is on for training
    model.train()

    return test_loss, accuracy



# Call to main function to run the program
if __name__ == "__main__":
    main()

