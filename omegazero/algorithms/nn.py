#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
import os

import config

artifacts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "artifacts")

class CustomDataset(Dataset):
    def __init__(self, state_data, policy_data, value_data):
        self.state_data = state_data
        self.policy_data = policy_data
        self.value_data = value_data

    def __len__(self):
        return len(self.state_data)

    def __getitem__(self, index):
        state = self.state_data[index]  # Shape (8, 8, 32)
        policy = self.policy_data[index]  # Shape (4096)
        value = self.value_data[index]  # Shape (1,)

        # Convert to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float32)
        policy = torch.tensor(policy, dtype=torch.float32)

        return state, policy, value

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 32, 8, 8)
        s = self.conv1(s)
        s = self.bn1(s)
        s = F.relu(s)
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(192, 8)
        self.fc2 = nn.Linear(8, 1)
        
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(64)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        
        # ([batch_size, 128, 8, 8]) to ([batch_size, 192])
        v = v.view(s.size(0), -1)

        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        s = self.conv1(s)
        s = self.bn1(s)
        p = F.relu(s) # policy head
        p = p.view(-1, 4096)
        p = self.logsoftmax(p).exp()
        return v, p
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        if not isinstance(s, torch.Tensor):
            s = torch.FloatTensor(s.astype(np.float64))
            if torch.cuda.is_available():
                s = s.contiguous().cuda()
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy*(1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
    
class NNManager():
    def __init__(self, player):
        self.player = player
        self.learning_rate = config.LEARNING_RATE

    @property
    def model_path(self):
        if self.player.is_best_player:
            return os.path.join(artifacts_path, "best_player.pth.tar")
        
        return os.path.join(artifacts_path, "new_player_nn_%s.pth.tar" % self.player.id)

    def learn(self, memory, iteration):
        net = ChessNet()
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate, betas=(0.8, 0.999))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,400], gamma=0.77)
        start_epoch = self.load_state(net, optimizer, scheduler)
        self.train(memory, net, optimizer, scheduler, start_epoch, iteration)

    def load_state(self, net, optimizer, scheduler):
        """ Loads saved model and optimizer states if exists """
        start_epoch, checkpoint = 0, None
        if os.path.isfile(self.model_path):
            checkpoint = torch.load(self.model_path)
        if checkpoint != None:
            if (len(checkpoint) == 1):
                net.load_state_dict(checkpoint['state_dict'])
                print("Loaded checkpoint model %s." % self.model_path)
            else:
                start_epoch = checkpoint['epoch']
                net.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("Loaded checkpoint model %s, and optimizer, scheduler." % self.model_path)    
        return start_epoch
    
    def predict(self, game_state_tensor):
        net = ChessNet()
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
            
        if os.path.isfile(self.model_path):
            checkpoint = torch.load(self.model_path)
            net.load_state_dict(checkpoint['state_dict'])

        net.eval()
        if cuda:
            net.cuda()
        with torch.no_grad():
            preds = net(game_state_tensor)
        return preds
    
    def train(self, memory, net, optimizer, scheduler, start_epoch, iteration):
        cuda = torch.cuda.is_available()
        net.train()
        criterion = AlphaLoss()
        
        tuple_data = memory.ltmemory_nparray
        state_data = tuple_data[0]
        policy_data = tuple_data[1]
        value_data = tuple_data[2]
        train_set = CustomDataset(state_data, policy_data, value_data)
        train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
        
        print("Starting training process...")
        update_size = len(train_loader)//10
        print("Update step size: %d" % update_size)
        for epoch in tqdm(range(start_epoch, config.NUM_EPOCHS)):
            total_loss = 0.0
            losses_per_batch = []
            for i,data in enumerate(train_loader,0):
                state, policy, value = data
                state, policy, value = state.float(), policy.float(), value.float()
                if cuda:
                    state, policy, value = state.cuda(), policy.cuda(), value.cuda()
                value_pred, policy_pred = net(state) # policy_pred = torch.Size([batch, 4672]) value_pred = torch.Size([batch, 1])
                loss = criterion(value_pred, value, policy_pred, policy)
                loss = loss/config.GRADIENT_ACC_STEPS
                loss.backward()
                clip_grad_norm_(net.parameters(), config.MAX_NORM)
                if (epoch % config.GRADIENT_ACC_STEPS) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                total_loss += loss.item()
                if i % update_size == (update_size - 1):    # print every update_size-d mini-batches of size = batch_size
                    losses_per_batch.append(config.GRADIENT_ACC_STEPS*total_loss/update_size)
                    total_loss = 0.0
            
            scheduler.step()
            if (epoch % 2) == 0:
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': net.state_dict(),\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                    }, self.model_path)
            '''
            # Early stopping
            if len(losses_per_epoch) > 50:
                if abs(sum(losses_per_epoch[-4:-1])/3-sum(losses_per_epoch[-16:-13])/3) <= 0.00017:
                    break
            '''
        print("Finished Training!")