
def check_command_line_arguments_for_train(in_arg):
    
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")   
        
    else:
        print(" Command Line Arguments:\n \n Data =", in_arg.dir,
              "\n",
              "\n Model:", in_arg.archi,
              "\n",
              "\n No of first hidden layers is:", in_arg.fl, 
              "\n",
              "\n No of second hidden layers is:", in_arg.shl,
              "\n",
              "\n Learning Rate:", in_arg.l,
              "\n",
              "\n Device:", in_arg.gpu,
              "\n",
              "\n Epoch:", in_arg.epoch,
              "\n",
             "\n Image:", in_arg.input,
              "\n",
              "\n Mapping file:", in_arg.output,
              "\n",
             "\n Print topk?:", in_arg.topk)
        
        
