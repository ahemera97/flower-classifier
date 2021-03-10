import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import numpy as np
import argparse
import json
from functions import args_paser_train, process_data, build_model, train_model, test_model, save_checkpoint

args=args_paser_train()

data_dir = args.data_dir
gpu = args.gpu
lr = args.lr
epochs = args.epochs
arch = args.arch
hidden_units = args.hidden_units
save_dir = args.save_dir

 
#processing data
trainloader, validloader, testloader , train_dataset = process_data(data_dir)
 
    
#build model
model = build_model(arch, hidden_units)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
device = torch.device('cuda' if gpu else 'cpu')


#train model 
train_model(model, device, epochs, trainloader,criterion, optimizer, validloader)

#test model
test_model(model, testloader, criterion, device)

#save check point
save_checkpoint(model, arch, epochs, optimizer, save_dir, train_dataset)



