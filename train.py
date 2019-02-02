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
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import signal
from   contextlib import contextmanager
import requests
import subprocess
import optparse
#===================================================================#
#workspace utilities
#workspace_utils saved code here Instead of downloading and saving the file
DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}
def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler
@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)
#====================================================================#
#Models
def select_model_structure(model):
    model_dict = {
     'alexnet' : [ models.alexnet(pretrained=True) , 9216  ],
     'vgg16'   : [ models.vgg16(pretrained=True), 25088 ],
     'densenet' : [models.densenet161(pretrained=True), 1024]
    }
    return (model_dict[model])
#####
#==================================================================#
def data_loader(dir,transformer):
    data       = datasets.ImageFolder(dir, transform=transformer)
    dataLoader = torch.utils.data.DataLoader(data,batch_size=64,shuffle=True)
    return (dataLoader)
####
#==================================================================#
def train_model(model,input_units,output_units,learning_rate,epochs,train_loaders,validation_loaders):
    stderr.write("the training model defaultly runs on cuda")
    #Freeze parameters
    for each_parameter in model.parameters():
        each_parameter.requires_grad = False
    #Feed-forward classifier
    NN_classifier =  nn.Sequential(OrderedDict([
                              ('dropout' , nn.Dropout(0.25)),
                              ('input_hidden_1', nn.Linear(input_units,1024)),
                              ('relu_1', nn.ReLU()),
                              ('hidden_1_hidden_2', nn.Linear(1024,512)),
                              ('relu_2', nn.ReLU()),
                              ('hidden_2_output', nn.Linear(512, output_units)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = NN_classifier
    loss_fun         = nn.NLLLoss()
    optimizer        = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.cuda()
    #######################
    epochs           = epochs
    step             = 0
    print_every      = 0
    model.to('cuda')
    #generator to iterator conversion
    with active_session():
        for e in range(epochs):
            train_loss            = 0
            for images,lables in iter(train_loaders):
                step+=1
                images,lables   = images.to('cuda'),lables.to('cuda')
                optimizer.zero_grad()
                y     = model.forward(images)
                error = loss_fun(y,lables)
                error.backward()
                optimizer.step()
                train_loss+=error.item()
                if not (step==5) : continue
                print_every+=step
                stderr.write("validationg at epoch %d  step %d"%(e+1,print_every))
                step                 = 0
                validation_loss      = 0
                accuracy             = 0
                #setting drop out to zero
                model.eval()
                n_valid_data = len(validation_loaders)
                for v_images, v_labels in iter(validation_loaders) :
                    optimizer.zero_grad()
                    v_images, v_labels = v_images.to('cuda:0'),v_labels.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():
                        y_hat     = model.forward(v_images)
                        validation_loss+=loss_fun(y_hat,v_labels)
                        #Accuracy
                        #pick top class
                        top_p, top_class = torch.exp(y_hat).topk(1,dim=1)
                        #equate dimessions , returns T/F
                        pr_label_match  = top_class == v_labels.view(*top_class.shape)
                        accuracy+=torch.mean(pr_label_match.type(torch.FloatTensor))
                #### end of data in validation set
                #reset the dropout probability
                model.train()
                stderr.write("Epoch: {}/{}.. ".format(e+1, epochs),
                      "steps %d"%print_every,
                      "Training Loss: {:.3f}.. ".format(train_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/n_valid_data),
                      "Validation Accuracy: {:.3f}".format(accuracy/n_valid_data))
            #### end of epochs
        ####
    #####
    return(model,model.classifier)
############
#===================================================================#
def save_model(model,train_dir,train_transforms,model_name,model_classifier):
    train_datasets     = datasets.ImageFolder(train_dir, transform=train_transforms)
    #TODO: Save the checkpoint
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'architecture': "%s"%model_name,
             'classifier': model_classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}
    torch.save(checkpoint,"Image_classifier_checkpoint.pth")
    stderr.write("trained model saved in current directory with name 'Image_classifier_checkpoint.pth' " )
######
#===================================================================#
def real_main():
    #Addiding commndline options
    usage  = "usage: %prog [options]"
    parser = optparse.OptionParser(usage)
    #reference file name
    parser.add_option( '-I', \
                       "--/Input_folder", \
                       type    = "str", \
                       help    = "path to folder that include the input data",
                       default = "")
    parser.add_option( '-m', \
                       "--model", \
                        type    = "str", \
                        help    = "architectures available from torchvision.models \
                       models vailable : alexnet,vgg16,densenet",
                       default = "vgg16" )

    parser.add_option( '-lr', \
                   "--learning_rate", \
                   type    = "float", \
                   help    = "hyperparameter learning rate default = 0.001",
                    default = 0.001)
    parser.add_option( '-o', \
                   "--number_of_outputs", \
                   type    = "int", \
                   help    = "number of outputs i.e. number of output types in the training data\
                             default = 102 for the flowers data",
                    default = 102)
    parser.add_option( '-e', \
                   "--number_of_epochs", \
                   type    = "int", \
                   help    = "number of epochs",
                   default = 8)
    # Parsing the arguments
    (options,args) = parser.parse_args()
    basePath             = options.Input_folder
    model_name           = "%s"%options.model
    model,input_units    = select_model_structure(model_name)
    lr                   = float(options.learning_rate)
    output_units         = int(options.number_of_outputs)
    epochs               = int(options.number_of_epochs)
    #Data_folders_for_current_data
    data_dir  = basePath + 'flowers'
    if(data_dir.count("tar.gz")>0) : system("tar -xvf %s"%data_dir)
    elif(data_dir.count("gz">0))   : system("gunzip %s"%data_dir)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    #test_dir  = data_dir + '/test'
    #####
   #Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize((224,224)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    #Using the image datasets and the trainforms, define the dataloaders
    train_loaders       =  data_loader(train_dir,train_transforms)
    validation_loaders  =  data_loader(valid_dir,test_transforms)
    #test_loaders       =  data_loader(test_dir,test_transforms)
    trained_model,model_classifier = train_model(model,input_units,output_units,lr,epochs,train_loaders,validation_loaders)
    save_model(trained_model,train_dir,train_transforms,model_name,model_classifier)
####
#===================================================================#
if ( __name__ == '__main__' ):
    real_main()
