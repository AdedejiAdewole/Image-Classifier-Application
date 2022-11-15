import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def build_model(arc, hidden1, hidden2, lr, device):

    if arc == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False    
       
        model.classifier = nn.Sequential(nn.Linear(25088, hidden1),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(hidden1, hidden2),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(hidden2, 102),
                           nn.LogSoftmax(dim=1))  
               
    elif arc == "densenet121":
        model = models.densenet161(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(nn.Linear(2208, hidden1),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(hidden1, hidden2),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(hidden2, 102),
                           nn.LogSoftmax(dim=1))
        
        
    elif arc == "densenet161":
        model = models.densenet161(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(nn.Linear(1024, hidden1),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(hidden1, hidden2),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(hidden, 102),
                           nn.LogSoftmax(dim=1))            
    else:
        print("wrong architecture entered")
        

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    model.to(device)
    
    return model, criterion, optimizer