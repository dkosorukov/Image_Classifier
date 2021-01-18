# Import python modules
import argparse
import json
from PIL import Image
import numpy as np


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def get_input_args():
            
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type=str, help='path of image to be classified')
    parser.add_argument('saved_model_path', type=str, default='checkpointP2.pth', help='path to saved model')
    parser.add_argument('--top_k', type=int, default=5, help='display top k probabilities')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to image name mapping json')
    parser.add_argument('--gpu', action="store_true", default='False', help='--gpu for GPU calculations, otherwise CPU is used')
    
    args = parser.parse_args()
        
    return args


def load_checkpoint(filepath):
    ''' Load pretrained model saved in filepath
        Load: Classifier, State Dictionary, Class to Index Disctionary
    '''
    
    checkpoint = torch.load(filepath, map_location=('cuda:0' if (torch.cuda.is_available()) else 'cpu'))
    arch = checkpoint['arch']
    model = getattr(models, arch)(pretrained=True)
    model.class_to_idx = checkpoint['class_mapping']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    # Freezing pretrained parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()    
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Resize the image to a given pixels of the shortest side
    shortest_side = 256
    image = Image.open(image_path)
    
    width, height = image.size
    aspect_ratio = width / height
    
    # Step 1 shrink iamge to shrotest side equal to sortest_side
    if width > height:
        new_height = shortest_side
        new_width = int(aspect_ratio * new_height)
    elif width < height:
        new_width = shortest_side
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = new_height = shortest_side
    size = (new_width, new_height)
    
    image.thumbnail(size, Image.LANCZOS)
    
    # Step 2 Center crop of the image to a given pixels square
    req_dimension = 224
    
    # Update dimensions of resized image after Step 1
    width, height = image.size
    left = (width - req_dimension)/2
    upper = (height - req_dimension)/2
    right = left + req_dimension
    lower = upper + req_dimension
        
    image = image.crop((left, upper, right, lower))
    
    # Step 3 Normalizing image
    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
    
    image = np.array(image)
    image = image/255
    image = ((image - mean) / sd)
    # Color chanel moved from 3rd to 1st dimension, retaining the other two
    image = np.transpose(image, (2, 0, 1))
    
    # Retrun Numpy array as required by the assignment, do not convert into tensor
    return image

def predict(image_path, model, gpu_flag, topk):
    ''' Predict the class (or topk classes) of an image using a trained deep learning model.
    '''
    
    # Activate CPU if so requested
    device = torch.device("cuda:0" if gpu_flag else "cpu")
    # If GPU is not available, reset back to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Switch off dropout
    
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.float()
    image = image.to(device)
    image = image.unsqueeze_(0)
    
    with torch.no_grad(): # Switch off gradient calculations
        
        logps = model.forward(image)
        ps = torch.exp(logps)
        # Get top k probabilities and indicies
        top_p, top_idx = ps.topk(topk, dim=1)
        
        # Invert ImageFolder's class_to_index dictionary so that provided index returns class (folder's name)
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        top_k_classes = [idx_to_class[idx] for idx in top_idx[0].tolist()]
          
    model.train()
    
    return top_p, top_k_classes
    
def main():
    # Command line arguments
    in_args = get_input_args()
    
    # Load category to image name dictionary
    with open(in_args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    
    # load model
    print("Saved checkpoint: ", in_args.saved_model_path) 
    model = load_checkpoint(in_args.saved_model_path)
    
    img_path = in_args.image_path
    print("Image path: ", img_path)
    topk = in_args.top_k
    print("Number of top predicted classes: ", topk)
    gpu_flag = in_args.gpu
    print("GPU flag: ", gpu_flag)
    
    # Run prediction model
    probs, predicted_label = predict(img_path, model, gpu_flag, topk)
    # Convert classes (folder names) into flower names
    predicted_name = [cat_to_name[label] for label in predicted_label]
    print("Predicted names: ", predicted_name)
    print("Prediction probabilities: ", probs[0].numpy())
    
    
if __name__ == "__main__":
    main()              
                                                 
    
    