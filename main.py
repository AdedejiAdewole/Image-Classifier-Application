from time import time, sleep
from check_results import *
from get_input_args import train_get_input_args
from datapreprocess import data_preprocess
from buildmodel import build_model
from train import train_model
from savemodel import save_model
from loadmodel import load_model
from predict import predict
import json 
import os
    
    
def main():
    
    start_time = time()
    
    in_arg = train_get_input_args()
    
    check_command_line_arguments_for_train(in_arg)
    
    train_data, trainloader, validloader, testloader = data_preprocess(in_arg.dir)
        
    model, criterion, optimizer = build_model(in_arg.archi, in_arg.fl, in_arg.shl, in_arg.l, in_arg.gpu)
    
    new_model = train_model(in_arg.epoch, trainloader, in_arg.gpu, optimizer, model, criterion, validloader)
    
    save_model(new_model, train_data, optimizer)
    
    loaded_model = load_model(in_arg.archi, '/home/workspace/ImageClassifier/model4.pth')
    
    json_file = in_arg.output
        
    with open(json_file, 'r') as f:    
        cat_to_name = json.load(f, strict = False)
    
    predict(in_arg.gpu, in_arg.input, loaded_model, cat_to_name, in_arg.topk)
       
    
    end_time = time()
    
    # TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":                      
    main()    
    
    
    
    
    