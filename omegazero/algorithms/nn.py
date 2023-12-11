#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
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
        state = self.state_data[index]  # Shape (8, 8, 33) # 32 channels for pieces, 1 channel for turn
        policy = self.policy_data[index]  # Shape (4096)
        value = self.value_data[index]  # Shape (1,)

        # Convert to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float32)
        policy = torch.tensor(policy, dtype=torch.float32)

        return state, policy, value

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(33, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 33, 8, 8)
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
        policy_error = -torch.sum(policy * torch.log(y_policy + 1e-8))
        total_error = value_error.view(-1).float() + policy_error
        return total_error
    
class NNManager():
    def __init__(self, episode):
        self.episode = episode
        self.learning_rate = config.LEARNING_RATE
        self.net = ChessNet()
        cuda = torch.cuda.is_available()
        if cuda:
            self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.8, 0.999))
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50,100,150,200,250,300,400], gamma=0.77)
        self.checkpoint = None
        self.start_epoch = 0
        saved_model_path = self.get_latest_model_path()
        if saved_model_path is not None and os.path.isfile(saved_model_path):
            print("=> loading checkpoint '{}'".format(saved_model_path))
            self.checkpoint = torch.load(saved_model_path)
            self.net.load_state_dict(self.checkpoint['state_dict'])
            if self.checkpoint != None:
                if not (len(self.checkpoint) == 1):
                    self.optimizer.load_state_dict(self.checkpoint['optimizer'])
                    self.scheduler.load_state_dict(self.checkpoint['scheduler'])

    def get_latest_model_path(self):
        files = [f for f in os.listdir(artifacts_path) if f.endswith(".pth.tar")]
        paths = [os.path.join(artifacts_path, basename) for basename in files]
        latest_model_path = max(paths, key=os.path.getctime, default=None)
        return latest_model_path

    @property
    def model_path(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(artifacts_path, f"new_player_nn_{timestamp}.pth.tar")
    
    def predict(self, game_state_tensor):
        with torch.no_grad():
            preds = self.net(game_state_tensor)
        return preds
    
    def save_loss_data(self, loss_history):
        columns = ['episode', 'epoch', 'loss']
        df = pd.DataFrame(loss_history, columns=columns)

        # check if loss_data.csv exists
        try:
            loss_data = pd.read_csv("loss_data.csv")
        except FileNotFoundError:
            loss_data = pd.DataFrame(columns=columns)

        loss_data = pd.concat([loss_data, df], ignore_index=True)
        loss_data.to_csv("loss_data.csv", index=False)
    
    def learn(self, memory):
        cuda = torch.cuda.is_available()
        self.net.train()
        criterion = AlphaLoss()
        if cuda:
            criterion.cuda()
        
        tuple_data = memory.ltmemory_nparray
        state_data = tuple_data[0]
        policy_data = tuple_data[1]
        value_data = tuple_data[2]
        
        state_data = torch.tensor(state_data, dtype=torch.float32)
        policy_data = torch.tensor(policy_data, dtype=torch.float32)
        value_data = torch.tensor(value_data, dtype=torch.float32)
        
        if cuda:
            state_data = state_data.to('cuda')
            policy_data = policy_data.to('cuda')
            value_data = value_data.to('cuda')
        
        train_set = CustomDataset(state_data, policy_data, value_data)
        train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

        loss_history = []
        
        print("Starting training process...")
        for epoch in tqdm(range(self.start_epoch, config.NUM_EPOCHS)):
            batch_loss_history = []
            for i,data in enumerate(train_loader,0):
                state, policy, value = data
                value_pred, policy_pred = self.net(state)
                loss = criterion(value_pred, value, policy_pred, policy)
                loss.backward()
                clip_grad_norm_(self.net.parameters(), config.MAX_NORM)
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_loss_history.append(loss.item())
            loss_history.append((self.episode, epoch, np.average(batch_loss_history)))
            self.scheduler.step()
            
        self.save_loss_data(loss_history)

        torch.save({
            'state_dict': self.net.state_dict(),\
            'optimizer' : self.optimizer.state_dict(),\
            'scheduler' : self.scheduler.state_dict(),\
        }, self.model_path)
        print("Finished Training!")