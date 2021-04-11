# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from submission import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

# Augmentation part, idea from "A Simple Framework for Contrastive Learning of Visual Representations"
# https://arxiv.org/pdf/2002.05709.pdf
 
s1 = 0.5
s2 = 0.75
s3 = 1
color_jitter1 = transforms.ColorJitter(0.8*s1, 0.8*s1, 0.8*s1, 0.2*s1)
color_jitter2 = transforms.ColorJitter(0.8*s2, 0.8*s2, 0.8*s2, 0.2*s2)
color_jitter3 = transforms.ColorJitter(0.8*s3, 0.8*s3, 0.8*s3, 0.2*s3)
rnd_color_jitter1 = transforms.RandomApply([color_jitter1], p=0.8)
rnd_color_jitter2 = transforms.RandomApply([color_jitter2], p=0.8)
rnd_color_jitter3 = transforms.RandomApply([color_jitter3], p=0.8)
rnd_gray = transforms.RandomGrayscale(p=0.2)

aug_transform1 = transforms.Compose([
    transforms.RandomResizedCrop((96,96), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
    rnd_color_jitter1,
    rnd_gray,
    transforms.ToTensor(),
])

aug_transform2 = transforms.Compose([
    transforms.RandomResizedCrop((96,96), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
    rnd_color_jitter2,
    rnd_gray,
    transforms.ToTensor(),
])

aug_transform3 = transforms.Compose([
    transforms.RandomResizedCrop((96,96), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
    rnd_color_jitter3,
    rnd_gray,
    transforms.ToTensor(),
])

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = CustomDataset(root='/dataset', split="train", transform=train_transform)
augset1 = CustomDataset(root='/dataset', split="train", transform=aug_transform1)
augset2 = CustomDataset(root='/dataset', split="train", transform=aug_transform2)
augset3 = CustomDataset(root='/dataset', split="train", transform=aug_transform3)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
augloader1 = torch.utils.data.DataLoader(augset1, batch_size=256, shuffle=True, num_workers=2)
augloader2 = torch.utils.data.DataLoader(augset2, batch_size=256, shuffle=True, num_workers=2)
augloader3 = torch.utils.data.DataLoader(augset3, batch_size=256, shuffle=True, num_workers=2)

net = get_model()
#net = torch.nn.DataParallel(net)
net = net.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

print('Start Training')

net.train()
for epoch in range(90):
    running_loss = 0.0
    for i, data in enumerate(trainloader): #Original training set
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('Original Training Set:[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
    for i, data in enumerate(augloader1): #After 1st augmentation
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('After 1st augmentation: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0            
    for i, data in enumerate(augloader2):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('After 2nd augmentation: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))            
            running_loss = 0.0     

    for i, data in enumerate(augloader3):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('After 3rd augmentation: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0               

print('Finished Training')


#validset = CustomDataset(root='/dataset', split="val", transform=train_transform)
#valLoader = torch.utils.data.DataLoader(validset, batch_size=256, shuffle=True, num_workers=2)

#with torch.no_grad():
#    net.eval()
#    running_loss = 0
#    for i, data in enumerate(valLoader):
#        inputs, labels = data
#        inputs, labels = inputs.cuda(), labels.cuda()
        

#        outputs = net(inputs)
#        loss = criterion(outputs, labels)
        
#        running_loss += loss.item()
#        if i % 10 == 9:    # print every 10 mini-batches
#            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
#            running_loss = 0.0   


os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "net_demo.pth"))

print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'net_demo.pth')}")
