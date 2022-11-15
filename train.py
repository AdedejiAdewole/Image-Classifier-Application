import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import json
                    

#function to train the model
def train_model(epochs, trainloader, device, optimizer, model, criterion, validloader):
    
    steps = 0
    running_loss = 0
    print_every = 30
    
    
    for epoch in range(epochs):

        for inputs, labels in trainloader:
            
            inputs, labels = inputs.to(device), labels.to(device)

            steps += 1      
            
            optimizer.zero_grad()        
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward() 
            optimizer.step()
            running_loss += loss.item() 
            if steps % print_every == 0:
                print("calculating epoch")
                test_loss = 0
                accuracy = 0    
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {test_loss/len(validloader):.3f}.. "
                        f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    return model 

    
    




