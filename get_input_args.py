import argparse


def train_get_input_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type = str, default = '/home/workspace/ImageClassifier/flowers/', help = 'Image Dataset')

    parser.add_argument('--archi', type = str, default = 'vgg16', help = 'Type of Model')

    parser.add_argument('--fl', type = int, default = 1024, help = 'No of first hidden layers')

    parser.add_argument('--shl', type = int, default = 512, help = 'No of first hidden layers')

    parser.add_argument('--l', type = float, default = 0.001, help = 'Learning Rate')

    parser.add_argument('--gpu', type = str, default = 'cuda', help = 'Device type')

    parser.add_argument('--epoch', type = int, default = 1 , help = 'No of Epochs')     
    
    parser.add_argument('--input', type = str, default =
                        '/home/workspace/ImageClassifier/flowers/test/2/image_05107.jpg', 
                        help = 'path to the folder of flower images')    
    
    parser.add_argument('--output', type = str, default = '/home/workspace/ImageClassifier/cat_to_name.json', 
                        help = 'path to the folder for mapping')
    
    parser.add_argument('--topk', type = str, default = 'yes' , help = 'Print the top classes')
    
    
    return parser.parse_args()


