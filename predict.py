from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json 
import os
from process_image import * 
import argparse
from torchvision import models

#function to make predictions, get the top classes and map the classes to their category names   
def predict(device, image_path, new_model, mapping, top_classes, topk = 5):
    
    new_model.eval()
    new_model.to(device)
    
    image = Image.open(image_path)
    image = process_image(image_path)
    
    image_tensor = torch.from_numpy(image).float().to(device)
    
    with torch.no_grad():
        in_image = torch.autograd.Variable(image_tensor)
        
    in_image = in_image.unsqueeze(0)
    
    
    output = new_model(in_image)
    top_p, top_label = torch.topk(output, topk)
    top_p = top_p.exp()
    top_p = top_p.to(device)
    top_p = top_p.cpu().data.numpy()[0]

    indices_to_class = {v: k for k, v in
                        new_model.class_to_idx.items()}
    
    top_label = top_label.to(device)
    top_label = top_label.cpu().data.numpy().tolist()[0]

    top_flowers_classes = [indices_to_class[x] for x in top_label]   
    
    flower_file = image_path.split('/')[-2]
    image_path = Image.open(image_path)    
    
    
    name = [mapping[top_flowers_classes[0]]]    
    
    print("")
    print(f"The probability of the prediction is: {np.max(top_p)}")
    print("")
    print(f"The Flower name is: {name}")      
    
    print("")
    print(f"The probabilities of the prediction are: {top_p}")
    
    
    if top_classes == "yes":    
        print("")
        print(f"The top K classes are: {top_flowers_classes}")    
        
        
    else: 
        None 




