##
# AVMixup Implementation
##

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
        self.pool3 = nn.MaxPool2d(4)
        self.linear = nn.Linear(80, 43)

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
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

gtsrb_dataset = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training')

batch_size = 128
dataset_loader = torch.utils.data.DataLoader(gtsrb_dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)

###
# Initial Training
###

model = Model().half().cuda()
criterion = nn.CrossEntropyLoss()
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-2)
max_epochs = 1
import time
from tqdm import tqdm
print(len(dataset_loader))
for epoch in range(max_epochs):
    start = time.time()
    for i, (x,y) in tqdm(enumerate(dataset_loader)):
        x = x.half().cuda()
        y = y.cuda()
        pred = model(x)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        break
    end = time.time()
    print('Epoch ' + str(epoch) + ': ' + str(end-start) + 's')


from advertorch.attacks import LinfPGDAttack

from losses import AlphaLoss

loss_fn = AlphaLoss(classes=43, params={'alpha' : 1.2})

adversary = LinfPGDAttack(
    model, loss_fn=loss_fn, eps=0.3,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

###
# Adversarial Training
###
print(len(dataset_loader))
for epoch in range(max_epochs):
    start = time.time()
    for i, (x,y) in tqdm(enumerate(dataset_loader)):
        x = x.half().cuda()
        y = y.cuda()
        adv_untargeted = adversary.perturb(x, y)
        # AV Mixup
        alpha = torch.rand(1).half().cuda()
        av_mix = alpha*x + (torch.ones(1).half().cuda() - alpha)*adv_untargeted
        pred = model(av_mix)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        break
    end = time.time()
    print('Epoch ' + str(epoch) + ': ' + str(end-start) + 's')

print('Done')