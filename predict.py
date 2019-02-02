#!/usr/bin/python
_author__                       = "yesesri_cherukuri"
__date_created__                = "02/1/2019"
#===================================================================#
#Imports here
import torch
from torch import nn,optim
from torchvision import datasets, transforms , models
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
from sys import stderr
from os import system
from system import argv
import json
#===================================================================#
def select_model_structure(model):
    model_dict = {
     'alexnet' : [ models.alexnet(pretrained=True) , 9216  ],
     'vgg16'   : [ models.vgg16(pretrained=True), 25088 ],
     'densenet' : [models.densenet161(pretrained=True), 1024]
    }
    return (model_dict[model])
#####
#===================================================================#
def load_trained_models(basePath):
    checkpoint   = torch.load(basePath)
    model        = select_model_structure(checkpoint['architecture'])
    for param in model.parameters(): param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier   = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return(model)
####
#===================================================================#
def process_image(image):
    image_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    img       = Image.open(image)
    PIL_image = image_transform(img)
    #np_image  = np.array( PIL_image  )
    return( PIL_image )
#####
#===================================================================#
def imshow(image, ax=None, title=None):
    if ax is None: fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax
#===================================================================#
def predict(image_path, model, n_k):
    #Predict the class (or classes) of an image using a trained deep learning model.
    model.eval()
    image = ( process_image(image_path) ).unsqueeze_(0).float()
    with torch.no_grad(): y_hat = model.forward(image.to('cuda:0'))
    #pick top class
    top_p, top_class = torch.exp(y_hat).topk(n_k)
    return(np.array(top_p.detach_())[0] , np.array(top_class.detach_())[0] )
####
#===================================================================#
def plot_figure(image_path,top_p,top_class,map_file):
    with open("%s"%map_file, 'r') as f: cat_to_name = json.load(f)
    imshow(process_image(image_path))
    flower_labels = []
    for n in top_class:
        n+=1
        flower_labels.append(cat_to_name['%s'%n])
    ####
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=top_p, y=flower_labels, color=sns.color_palette()[0])
    plt.show()
#####
#===================================================================#
def real_main():
    #usage : python predict.py  image_path model_file number_top_class
    image_path            = "%s"%argv[1]
    checkpoint_file_path  = "%s"%argv[2]
    Json_map_file         = "%s"%argv[3]
    number_top_class      = "%d"%argv[4]
    ########################################################
    #Example :
    #image_path            = "flowers/test/1/image_06743.jpg"
    #checkpoint_file_path  = "Image_classifier_checkpoint.pth"
    #number_top_class      = 5
    ########################################################
    model           = load_trained_models(checkpoint_file_path)
    top_p,top_class = predict(path,model,number_top_class)
    plot_figure(image_path,top_p,top_class,Json_map_file)
#####
#===================================================================#
if ( __name__ == '__main__' ):
    real_main()
