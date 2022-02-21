from dataset import *

###
# Model Checking
###

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3,3))
        self.conv2 = nn.Conv2d(5,5, (3,3))
        self.pool1 = nn.MaxPool2d(4)
        self.conv3 = nn.Conv2d(5, 10,(3,3))
        self.conv4 = nn.Conv2d(10, 15, (3,3))
        self.pool2 = nn.MaxPool2d(4)
        self.conv5 = nn.Conv2d(15, 15, (3,3))
        self.conv6 = nn.Conv2d(15, 20, (3,3))
        self.pool3 = nn.MaxPool2d(2)
        self.linear = nn.Linear(320, 43)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = self.pool3(x)

        x = self.linear(x.flatten())
        return x

gtsrb_dataset = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training')

batch_size = 64
dataset_loader = torch.utils.data.DataLoader(gtsrb_dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)

model = Model()

max_epochs = 1

for epoch in range(max_epochs):

    for i, (x,y) in enumerate(dataset_loader):
        print(x.size())
        print(y.size())
        break 