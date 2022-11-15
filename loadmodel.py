from torch import optim
import torch
from torchvision import models

def load_model(arch, model_path):     
    
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        
    elif arch == "densenet121":
        model = models.densenet161(pretrained=True)
        
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)    
        
    else:
        print("wrong architecture entered")    
           
    
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model_loaded = torch.load(model_path)
          
          
    model.classifier = model_loaded['classifier']
    model.load_state_dict(model_loaded['state_dict']) 
    model.class_to_idx = model_loaded['class_to_idx']
    
    return model




