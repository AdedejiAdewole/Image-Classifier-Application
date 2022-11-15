from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

#function to process the image to make prediction
def process_image(image):
    img = Image.open(image)
    img.load()
    

    #Resize the image
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
        

    #Crop image
    size = img.size
    img = img.crop((size[0]//2 -(224/2),
                     size[1]//2 - (224/2),
                     size[0]//2 +(224/2),
                     size[1]//2 + (224/2) 
                    ))
    
    #Normalize image
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    img = (img - mean)/std
    
    final_image = img.transpose((2, 0, 1))
                                
    
    return final_image