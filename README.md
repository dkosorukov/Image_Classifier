# AI Programming with Python Project

Jupyter Notebook for an image classifier built with PyTorch.
Notebook is converted into a command line application.
Application includes a train part and a predict part.

## Project Components
Package includes a Jupyter Notebook and command line applications, one (train.py) to train the network and one (predict.py) to run prediction.
Training image dataset includes 102 flower categories and comes from the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

## Packages
Project uses python 3, argparse, json, torch, torchvision, numpy, random, matplotlib, PIL, glob, and jupyter nootebook

## Model
Model uses a pretrained network. Only classifier is retrained to a specific image set. Pretrained model is either vgg13 or vgg16, the latter is a default setting. Classifier includes two hidden linear layers. The first one includes ReLU and Dropout, the second includes LogSoftmax. Model details:
```
 model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, output_units),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
 ```
Reasonably good results (accuracy of 0.889) are observed at the number of hidden layers of 1024, dropout rate of 0.2, epoches of 5, batches of 64, and learning rate of 0.001. Training loss in this scenario is 0.865 and test loss is 0.419.

## License
myanswers.py is a public domain work, dedicated using [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Feel free to do whatever you want with it.
