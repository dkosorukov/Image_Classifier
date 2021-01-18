# Imports python modules
import argparse
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models



def get_input_args():
    
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder with default value 'flowers'
      2. Checkpoint Folder as --save_dir
      3. CNN Model Architecture as --arch with default value 'vgg'
      Network hyperparameters:
      4. Learning rate as --learning_rate with default value of 0.003
      5. Number of hidden units as --hidden_units with default value of 512
      6. Number of epochs as --epochs with default value of 3
      7. Use of GPU as --gpu with default value of cpu
      
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
        
    # Creates parse 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory', type=str, default='flowers', help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='', help='path to save checkpoint file')
    parser.add_argument('--arch', type=str, choices=['vgg13', 'vgg16'], default = 'vgg16', help='model: vgg13, vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate, typically 0.01 - 0.0001')
    parser.add_argument('--hidden_units', type=int, default=1024, help='# hidden units of next to the last layer')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    #parser.add_argument('--gpu', type=bool, default='True', help='use GPU for training: 1 GPU, 0 CPU')
    parser.add_argument('--gpu', action="store_true", default='False', help='--gpu for GPU calculations, otherwise CPU is used')

    args = parser.parse_args()
                                                         
    return args

def save_checkpoint(filepath, model, optimizer, arch, epochs, learning_rate, hidden_units, train_dict):
    model.class_to_idx = train_dict
    device = torch.device("cpu")
    model.to(device)
    checkpoint = {
                  'arch': arch,
                  'epochs': epochs,
                  'lr': learning_rate,
                  'hidden_units': hidden_units,
                  'class_mapping': train_dict,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer,
                  'optimizer_dict': optimizer.state_dict()
                   }
    
    torch.save(checkpoint, filepath)


def main():
    # Command line arguments
    in_args = get_input_args()
    
    # Other inputs
    # Training batch size
    batch_size = 64
    # Number of classes
    output_units = 102
                                                     
    # Loading and transforming image data
    data_dir = in_args.data_directory
    print('Data directory: ', data_dir)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Transformers of image data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                      ])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                         ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_dict = train_data.class_to_idx
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
           
    # Loading pretrained model
    arch = in_args.arch
    print('Model arch: ', arch)
    
    model = getattr(models, arch)(pretrained=True)

    # Freezing model's pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier
    hidden_units = in_args.hidden_units
    print('Number of hidden units: ', hidden_units)

    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, output_units),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    learning_rate = in_args.learning_rate
    print('Learning rate: ', learning_rate)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Build and train network

    # Activate CPU if so requested
    print('GPU activation: ', in_args.gpu)
    device = torch.device("cuda:0" if in_args.gpu else "cpu")
    # If GPU is not available, reset to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = in_args.epochs
    print('Epochs: ', epochs)
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # Reset optimizer gradient from prior batch
            optimizer.zero_grad()

            # Model run
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0

                model.eval() # Switch off dropout

                with torch.no_grad(): # Switch off gradient calculations
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0

                model.train() # Switch on dropout

    # Save checkpoint
    save_checkpoint('checkpointP2.pth', model, optimizer, arch, epochs, learning_rate, hidden_units, train_dict)

if __name__ == "__main__":
    main()


              
                                                 
    
    