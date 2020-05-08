#utility.py

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
import os


# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):

        checkpoint = torch.load(filepath) 
        model = Network(checkpoint['input_size'],
                                checkpoint['output_size'],
                                checkpoint['hidden_layers'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    w,h = img.size
    if w<h:
        img = img.resize((256, int(256*(h/w))))
    else: img = img.resize((int(256*(w/h)), 256))
    w,h = img.size
    img = img.crop(((w - 224)/2, (h - 224)/2, (w + 224)/2, (h + 224)/2))
    np_image = np.array(img)
    np_image[0] = (np_image[0] - 0.485)/0.229
    np_image[1] = (np_image[1] - 0.456)/0.224
    np_image[2] = (np_image[2] - 0.406)/0.225
    np_image = (np_image.transpose((2, 0, 1)))/255
    np_image = np_image[np.newaxis,:]
    img = torch.from_numpy(np_image)
    img = img.float()
    return img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)

    model.eval()
    img = img.resize_(img.size()[0], 25088)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = model.forward(img.to(device))    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)



#Network Definition
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):       
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        return F.log_softmax(x, dim=1)
