#Attempted on April_17th

#import part
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from submission import get_model

#parser part
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()
#train composition operations
train_transform = transforms.Compose([
    transforms.ToTensor(),
])
#trainset and trainloaders
trainset = CustomDataset(root='./dataset', split="train", transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

#net = get_model().cuda()
net = get_model()

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

print('Start Training')

#training part
net.train()
#hyperparameters
epoch_num = 80
h_dim = 400
z_dim = 20
#batch_size = 256
#lr = 0.05
#image_size = 784

#variational encoder
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    #encoder
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
        
    #random latent variable
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        mu_eps_std =mu + eps * std
        return mu_eps_std
        
    #decoder
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
        
    #Forward: from ecoder to decoder
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
        
#VAE_model = VAE().to(device)
VAE_model = VAE()
optimizer = torch.optim.Adam(VAE_model.parameters(), lr = 0.05)
    

#
for epoch in range(epoch_num):
    running_loss = 0.0
    for i, data in enumerate(trainloader): # data is the list of [inputs, labels]
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        
        inputs = inputs.view(-1, image_size)
        inputs_reconst, mu, log_var = VAE_model(inputs)
        
        reconst_loss = F.binary_cross_entropy(inputs_reconst,inputs, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        
        #outputs = net(inputs)
        #loss = criterion(outputs, labels)
        outputs = VAE_model(inputs)
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('Training Set: [%d, %5d] | loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

checkPointDir = './checkPoint'

os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "net_demo.pth"))

print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'net_demo.pth')}")
