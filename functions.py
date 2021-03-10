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



def args_paser_train():
    paser = argparse.ArgumentParser(description='trainer file')
    paser.add_argument('data_dir', type=str, default='flowers', help='dataset directory')
    paser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    paser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    paser.add_argument('--epochs', type=int, default=5, help='num of epochs')
    paser.add_argument('--arch', type=str, default='vgg16', help='choose between vgg16 and densenet121')
    paser.add_argument('--hidden_units', type=int, default=512, help='hidden units for layer')
    paser.add_argument('--save_dir', type=str, default='.', help='save train model to a file')
    args = paser.parse_args()
    return args

def args_paser_predict():
    paser = argparse.ArgumentParser(description='predict file')
    paser.add_argument('image_path', type=str, default = 'flowers/test/1/image_06743.jpg',help='Path to image, e.g., "flowers/test/1/image_06735.jpg"')
    paser.add_argument('checkpoint', type=str, default = '.', help='path to check point folder, e.g.,"assets/checkpoint.pth"')
    paser.add_argument('--top_k', type=int, default = 3,help='number of the top classes to show, e.g., 5')
    paser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    paser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to the mapping of categories file')
    args = paser.parse_args()
    return args



def process_data(data_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),             transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64)
    testloader  = torch.utils.data.DataLoader(test_dataset, batch_size = 64)
    
    return trainloader, validloader, testloader , train_dataset        


def build_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input = 25088
        print('train vgg16')
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        input = 1024
        print('train densenet121')
    else:
        print('only vgg16 and densenet121 are available')
        model = models.vgg16(pretrained = True)
        input = 25088
        print('train vgg16')
    
    
    for param in model.parameters():
        param.requires_grad =False
    classifier = nn.Sequential(nn.Linear(input, hidden_units), nn.ReLU(), nn.Dropout(0.2),nn.Linear(hidden_units, 102), nn.LogSoftmax(dim=1))
    model.classifier = classifier
    return model

def train_model(model, device, epochs, trainloader,criterion, optimizer, validloader):

    model.to(device);
    with active_session():
        epochs = epochs
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
        
                inputs, labels = inputs.to(device), labels.to(device)
        
                optimizer.zero_grad()
        
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                    
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                    
                            test_loss += batch_loss.item()
                    
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(validloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()
    print('\ntraining complete')
            
            
def test_model (model, testloader, criterion, device):
    # TODO: Do validation on the test set
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
                    
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                    
            test_loss += batch_loss.item()
                    
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")


def save_checkpoint(model, arch, epochs, optimizer, save_dir, train_dataset):
    # TODO: Save the checkpoint 
    model.to('cpu')
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'model': model,
                  'epochs': epochs + 1,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print('\nsaved')
    

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    
    return model

def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    width, height = img.size
    img = img.resize((256, int(256*(height/width))) if width < height else (int(256*(width/height)), 256))
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    img = np.array(img) / 255
    img = img.transpose((2, 0, 1))
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    

    return img


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)
    model.to(device)
    
    with torch.no_grad():
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.type(torch.FloatTensor)
        img = img.to(device)
    

    logps = model.forward(img)
    ps = torch.exp(logps)
    
    top_p, top_class = ps.topk(top_k, dim=1)
    probs = [float(prob) for prob in top_p[0]]
    class_map = {v: k for k, v in model.class_to_idx.items()}
    classes = [class_map[int(k)] for k in top_class[0]]
    
    return probs, classes


def check_result(image_path, model, cat_to_name):
    # TODO: Display an image along with the top 5 classes
    img = Image.open(image_path)
    probs, classes = predict(image_path, model)
    fig = plt.figure(figsize= [5,10])
    ax=plt.subplot(2, 1, 1)
    x= process_image(img)
    image =imshow(x,ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])

    classes_list =[cat_to_name[c] for c in classes]
    plt.title(classes_list[0])   ;
    plt.subplot(2,1,2)

    plt.barh(range(len((classes_list))), probs)
    plt.yticks(range(len(classes_list)),classes_list);
    plt.gca().invert_yaxis()
