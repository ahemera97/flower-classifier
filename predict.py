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
from functions import args_paser_predict, load_checkpoint, predict

args=args_paser_predict()
image_path = args.image_path
checkpoint = args.checkpoint + 'checkpoint.pth'
top_k = args.top_k
gpu = args.gpu
category_names = args.category_names

with open(category_names , 'r') as f:
    cat_to_name = json.load(f)

    
model = load_checkpoint(checkpoint)
device = torch.device('cuda' if gpu else 'cpu')
probs, classes = predict(image_path, model, top_k, device)

classes_list =[cat_to_name[c] for c in classes]

print("\nFlower name (probability): ")
print("---")
for i in range(len(probs)):
    print(f"{classes_list[i]} ({round(probs[i], 3)})")
print("")

