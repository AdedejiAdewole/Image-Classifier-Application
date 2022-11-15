import torch
from torch import nn
from torch import optim

def save_model(new_model, train_data, optimizer):
    
    new_model.class_to_idx = train_data.class_to_idx
    torch.save({'model_name': 'model_based_on_vgg',
                'state_dict': new_model.state_dict(), 
                'classifier': new_model.classifier,
                'optimizer_state_dict': optimizer.state_dict,
                'class_to_idx': new_model.class_to_idx}, 
                'ImageClassifier/model4.pth')