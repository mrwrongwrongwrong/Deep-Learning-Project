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

os.system("mkdir /tmp/wl2337/newlyLabeled")
os.system("mkdir /tmp/wl2337/givenLabeled")
os.system("echo make directary?")
os.system("ls /tmp/wl2337")
os.system("echo Anything there?")




os.system('''
        IDX=0
        while read p; do
            cp /dataset/unlabeled/$p /tmp/wl2337/newlyLabeled/${IDX}.png
            IDX=`expr $IDX + 1`
        done < movedFileNames.txt
        echo $IDX
        ''')


os.system('''
        IDX=0
        while read p; do
            cp /dataset/unlabeled/$p /tmp/wl2337/givenLabeled/${IDX}.png
            echo $p
            IDX=`expr $IDX + 1`
        done < request_10.csv
        echo $IDX
        ''')

os.system("cp newlyLabeled.pt /tmp/wl2337/newlyLabeled_label_tensor.pt")
os.system("cp label_10.pt /tmp/wl2337/givenLabeled_label_tensor.pt")
os.system("ls /tmp/wl2337")










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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=2)
augloader1 = torch.utils.data.DataLoader(augset1, batch_size=1024, shuffle=True, num_workers=2)
augloader2 = torch.utils.data.DataLoader(augset2, batch_size=1024, shuffle=True, num_workers=2)
augloader3 = torch.utils.data.DataLoader(augset3, batch_size=1024, shuffle=True, num_workers=2)

newly_trainset = CustomDataset(root='/tmp/wl2337/', split="newlyLabeled", transform=train_transform)
newly_augset1 = CustomDataset(root='/tmp/wl2337/', split="newlyLabeled", transform=aug_transform1)
newly_augset2 = CustomDataset(root='/tmp/wl2337/', split="newlyLabeled", transform=aug_transform2)
newly_augset3 = CustomDataset(root='/tmp/wl2337/', split="newlyLabeled", transform=aug_transform3)
newly_trainloader = torch.utils.data.DataLoader(newly_trainset, batch_size=1024, shuffle=True, num_workers=2)
newly_augloader1 = torch.utils.data.DataLoader(newly_augset1, batch_size=1024, shuffle=True, num_workers=2)
newly_augloader2 = torch.utils.data.DataLoader(newly_augset2, batch_size=1024, shuffle=True, num_workers=2)
newly_augloader3 = torch.utils.data.DataLoader(newly_augset3, batch_size=1024, shuffle=True, num_workers=2)

given_trainset = CustomDataset(root='/tmp/wl2337/', split="givenLabeled", transform=train_transform)
given_augset1 = CustomDataset(root='/tmp/wl2337/', split="givenLabeled", transform=aug_transform1)
given_augset2 = CustomDataset(root='/tmp/wl2337/', split="givenLabeled", transform=aug_transform2)
given_augset3 = CustomDataset(root='/tmp/wl2337/', split="givenLabeled", transform=aug_transform3)
given_trainloader = torch.utils.data.DataLoader(given_trainset, batch_size=1024, shuffle=True, num_workers=2)
given_augloader1 = torch.utils.data.DataLoader(given_augset1, batch_size=1024, shuffle=True, num_workers=2)
given_augloader2 = torch.utils.data.DataLoader(given_augset2, batch_size=1024, shuffle=True, num_workers=2)
given_augloader3 = torch.utils.data.DataLoader(given_augset3, batch_size=1024, shuffle=True, num_workers=2)



net = get_model()
checkPointDir = './model6.pth'
checkpoint = torch.load(checkPointDir)
net.load_state_dict(checkpoint)
net = net.cuda()


#net = get_model()
#net = torch.nn.DataParallel(net)
#net = net.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

print('Start Training')

net.train()
for epoch in range(50):
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
    for i, data in enumerate(newly_trainloader): #Original newly_training set
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        #print("inputs shape:",inputs.shape)
        #print("labels shape:",labels.shape)
        #print(labels)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('Original Training Set on newlyLabeled:[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
    for i, data in enumerate(newly_augloader1): #After 1st augmentation on newlyLabeled
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
            print('After 1st augmentation on newlyLabeled: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0            
    for i, data in enumerate(newly_augloader2):
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
            print('After 2nd augmentation on newlyLabeled: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))            
            running_loss = 0.0     

    for i, data in enumerate(newly_augloader3):
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
            print('After 3rd augmentation on newlyLabeled: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0                       

    for i, data in enumerate(given_trainloader): #Original given_training set
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        #print("inputs shape:",inputs.shape)
        #print("labels shape:",labels.shape)
        #print(labels)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('Original Training Set on givenLabeled:[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
    for i, data in enumerate(given_augloader1): #After 1st augmentation on givenLabeled
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
            print('After 1st augmentation on givenLabeled: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0            
    for i, data in enumerate(given_augloader2):
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
            print('After 2nd augmentation on givenLabeled: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))            
            running_loss = 0.0     

    for i, data in enumerate(given_augloader3):
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
            print('After 3rd augmentation on givenLabeled: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
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
