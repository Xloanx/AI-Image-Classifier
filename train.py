#train.py



# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import argparse
from utility import Network
import os
from workspace_utils import keep_awake

parser = argparse.ArgumentParser(description='Train AI image Classifier')
parser.add_argument('data_dir', type=str, default='flowers', help='Directory for the input data/image')
parser.add_argument('--save_dir', type=str, default='check_dir', help='Directory to save checkpoint')
parser.add_argument('--arch', type=str, default='vgg16', help='Architecture of pretrained net')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--hidden_units', type=int, default=4096, help='Number of units for the hidden layer')
parser.add_argument('--epochs', type=int, default=20, help='Number of times to run the datasets')
parser.add_argument('--gpu', type=bool, default=True, help='Whether or not to use gpu or cpu')
args = parser.parse_args()


#images drx
if os.path.exists(args.data_dir) and os.path.isdir(args.data_dir):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
else:
    print("The specified directory for input does not exist")




# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                      transforms.RandomResizedCrop(224), 
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229,0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

data_transforms = [train_transforms, valid_transforms, test_transforms]



# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=data_transforms[0])
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms[1])
test_data = datasets.ImageFolder(test_dir, transform=data_transforms[2])
image_datasets = [train_data, valid_data, test_data]



# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
dataloaders = [trainloader, validloader, testloader]


############################################TODO: Build and train your network
for i in keep_awake(range(5)):
    #Loading the pre-trained net
    if args.arch == "vgg":
        model = models.vgg16(pretrained=True)
        mod_classifier_input = 25088
            # Freeze parameters so we don't backprop through them
        for parameter in model.parameters():
            parameter.requires_grad=False
    elif args.arch == "densenet":
        model = models.densenet161(pretrained=True)
        mod_classifier_input = 2208
            # Freeze parameters so we don't backprop through them
        for parameter in model.parameters():
            parameter.requires_grad=False
    else:
        print("This app does not train with your selected architecture. Select either VGG or Densenet")


    # Hyperparameters for our network
    classifier_input = mod_classifier_input
    output_size = 102 #Number of flower categories in the dataset
    hidden_layer = [args.hidden_units]

    #Network definition with parameters
    newModel = Network(classifier_input, output_size, hidden_layer, drop_p=0.5)

    # Replace default classifier with new classifier
    model.classifier = newModel

    # Find the device availableto use using torch library
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    # Move model to the device specified above
    model.to(device)

    #Train the network
    # Criterion and optimizer definition
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    epochs = args.epochs
    for epoch in range(epochs):
        training_loss = 0
        validation_loss = 0
        accuracy = 0
        model.train()  # Training the model
        ("Training commenced for epoch {}".format(epoch+1))
        loop_count = 0
        for inputs, labels in iter(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()                               #optimizers cleared
            output = model.forward(inputs)                      #Forward pass        
            trloss = criterion(output, labels)                    #calc loss
            trloss.backward()                                     #Gradients calculation via backprops
            optimizer.step()                                    #Parameter adjustment
            training_loss += trloss.item()*inputs.size(0)         #Increment training loss with new loss
            loop_count += 1                                      #Increment loop count
            print(".",end="")                                    #progress tracker for training

        model.eval()                                                #Do validation on the validation set
        print("\nValidation commenced")
        loop_count = 0
        with torch.no_grad():                   # No calculation of gradients
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model.forward(inputs)
                valloss = criterion(output, labels)
                validation_loss += valloss.item()*inputs.size(0)                   #Increment validation loss with new loss
                output = torch.exp(output)                                         #reverse log for %
                top_p, top_class = output.topk(1, dim=1)                           #Azimuth of output 
                equals = top_class == labels.view(*top_class.shape)                #Qty correct classes
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()      #get running accuracy for the epoch
                loop_count += 1
                print(".",end="")                                                  #progress tracker for eval
        # Average details for the entire epoch
        training_loss = training_loss/len(trainloader.dataset)
        validation_loss = validation_loss/len(validloader.dataset)
        accuracy = accuracy/len(validloader)
        # Print out the information
        print('\nDetails for this Epoch')
        print('Epoch Number: {} \tAccuracy: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(epoch+1, accuracy, training_loss, validation_loss))


    # Save the checkpoint
    checkpoint = {'input_size': classifier_input,
                  'output_size': output_size,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'state_dict': model.classifier.state_dict(),
                    'optimizer':optimizer.state_dict()
                }

    if os.path.exists(args.save_dir) and os.path.isdir(args.save_dir):
        if torch.save(checkpoint, os.path.join(args.save_dir,'checkpoint.pth')):
            print("Checkpoint saved successfully at {}".format( os.path.join(args.save_dir,'checkpoint.pth')))
    else:
        os.mkdir(args.save_dir)
        if torch.save(checkpoint, os.path.join(args.save_dir,'checkpoint.pth')):
            print("Checkpoint saved successfully at {}".format( os.path.join(args.save_dir,'checkpoint.pth')))
