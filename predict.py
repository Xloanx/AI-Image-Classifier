#predict.py


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
from utility import load_checkpoint
from utility import process_image
from utility import imshow
from utility import predict
import json
import argparse
from workspace_utils import keep_awake

parser = argparse.ArgumentParser(description='Predicts Flower name from an image')
parser.add_argument('input_image_path', type=str, help='path to the image to be classified')
parser.add_argument('checkpoint', type=str, help='path to fetch the saved checkpoint')
parser.add_argument('--top_k', type=int, default=3, help='top K most likely cases')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='images mapping to real names')
parser.add_argument('--gpu', type=bool, default=True, help='Whether or not to use gpu or cpu')
args = parser.parse_args()


with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)                                          #read the json file into a dict

img = args.input_image_path
model = load_checkpoint(args.checkpoint)
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
probabilities, classes = predict(img, model.to(device),args.top_k)
category_names = [cat_to_name[str(cl)]for cl in classes]
for name, prob in zip(category_names, probabilities):
    print("The Model predicts {} with a probability of {}%\n".format(name,int(prob*100)))
