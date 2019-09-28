import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

# Loading the classifier models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_curr = os.path.dirname(os.path.abspath(__file__))
checkpoint_path_a = os.path.join(path_curr, "classifier_animals1.pth")
#checkpoint_path_a = os.path.join(path_curr, "classifier_habitats.pth")

chpt_a = torch.load(checkpoint_path_a)        
model_a = torchvision.models.resnet18(pretrained=True)
for param in model_a.parameters():
    param.requires_grad = False
        
model_a.class_to_idx = chpt_a['class_to_idx']
idx_to_class_a = {val: key for key, val in model_a.class_to_idx.items()}

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_a.fc.in_features
model_a.fc = nn.Linear(num_ftrs, 38)    
model_a = model_a.to(device)
model_a.load_state_dict(chpt_a['state_dict'])


# Process an image before passing it to the classifier for prediction
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image



# Predict the class of an image using a trained and loaded model.
def predict_image(image_path, model):
    
    # Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)
    
    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    
    model.eval()
    inputs = Variable(img).to(device)
    ps = torch.exp(model.forward(inputs))
    index = ps.data.numpy().argmax()
    return index


pred = predict_image('C:/Users/DELL/Downloads/elephant.jpg', model_a)
print("The Predicted Label is {}".format(idx_to_class_a[pred]))
x= "{}" .format(idx_to_class_a[pred])
print(x)