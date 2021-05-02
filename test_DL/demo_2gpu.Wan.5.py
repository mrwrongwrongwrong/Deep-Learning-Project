# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from submission import *

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

def trainLabeledImage(net, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    print('Start Training')

    net.train()
    for epoch in range(10):
    # for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # if i == 1:
            #     break
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels, idNum = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

    checkPointDir = './checkPoint'
    # os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(checkPointDir, exist_ok=True)
    # torch.save(net.module.state_dict(), os.path.join(args.checkpoint_dir, "net_demo.pth"))
    torch.save(net.module.state_dict(), os.path.join(checkPointDir, "net_demo.pth"))

    # print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'net_demo.pth')}")
    print(f"Saved checkpoint to {os.path.join(checkPointDir, 'net_demo.pth')}")
#################################################################3
def addIndexToTrainData(trainset):
    returnList = []
    for iC, (x, y) in enumerate(trainset):
        if iC % 1000 == 0:
            print(">>>>>>>>", iC)
        returnList.append((x, y, iC))
    return returnList
# ==========================================================================
# ==========================================================================
from submission import get_model, eval_transform, team_id, team_name, email_address

trainset = CustomDataset(root='./dataset', split="train", transform=train_transform)
# trainset = addIndexToTrainData(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)


net = get_model().cuda()
net = torch.nn.DataParallel(net)
net = net.cuda()
# trainLabeledImage(net, trainloader)
#
unLabeledSet = CustomDataset(root='./dataset', split="unlabeled", transform=train_transform)
xxx = addIndexToTrainData(unLabeledSet)
# unLabeledLoader = torch.utils.data.DataLoader(unLabeledSet, batch_size = 10, shuffle=True, num_workers = 2)
#
# net = get_model()
# checkPointDir = './checkPoint/net_demo.pth'
# # checkpoint = torch.load(args.checkpoint_path)
# checkpoint = torch.load(checkPointDir)
# net.load_state_dict(checkpoint)
# net = net.cuda()
#
# net.eval()
# correct = 0
# total = 0
# trainErr = 0.01
# with torch.no_grad():
#     newlyLabeledList = []
#     request_label_list = []
#
#     while(trainErr < 1):
#         for iC, data in enumerate(unLabeledLoader):
#             # for data in evalloader:
#             if iC == 9:
#                 break
#             images, labels = data
#             images = images.cuda()
#             labels = labels.cuda()
#             outputs = net(images)
#             #Smallest Margin Uncertainty
#             margin = torch.max(outputs, 1)[0] - torch.min(outputs, 1)[0]
#             minMarginPosition = torch.argmin(margin)
#             pickedUnlabelled = images[minMarginPosition]
#             _, predicted = torch.max(outputs.data, 1)
#
#             tempTouple = (pickedUnlabelled, predicted[minMarginPosition])
#             newlyLabeledList.append(tempTouple)
#
#             maxMarginPosition = torch.argmax(margin)
#             requsetlabels = images[maxMarginPosition]
#             temp_request = (requestlabels, predicted[maxMarginPosition])
#             request_label_list.append(temp_request)
            # print(tempTouple)
                # print("<<<<<<<<<<<<", predicted[minMarginPosition], minMarginPosition)

        # trainErr += 1
# newlyLabeledLoader = torch.utils.data.DataLoader(newlyLabeledList, batch_size=3, shuffle=True, num_workers=2)
# print("====>", trainset)
# for iC, (aa, bb) in enumerate(trainset):
#     if iC == 1:
#         break
#     print("<<====>", aa.shape, bb)
# print(f"Team {team_id}:
# {team_name} Accuracy: {(100 * correct / total):.2f}%")
# team_id, team_name, email_address = "dl10", "asdf", "asdf"
# print(f"Team {team_id}: {team_name} Accuracy: {(100 * correct / total):.2f}%")
# print(f"Team {team_id}: {team_name} Email: {email_address}")
