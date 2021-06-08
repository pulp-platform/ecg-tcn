#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import torch
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
def train(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    input_channels = 1
    seq_length = 140
    train_loss = Metric('train_loss')
    minimum = 0
    maximum = 0
    for local_batch, local_labels in train_loader:
        if(local_batch.max()>maximum):
            maximum = local_batch.max()
        if(local_batch.min()<minimum):
            minimum = local_batch.min()
    minimum = minimum.numpy()
    maximum = maximum.numpy()
    with tqdm(total=len(train_loader),
          desc='Train Epoch     #{}'.format(epoch + 1),
          disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            X = data.numpy()
            target = target
            X_std = (X - minimum) / (maximum - minimum)
            X_scaled = X_std * (1 - 0) + 0
            data = torch.from_numpy(X_scaled)
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            data = data.view(-1, input_channels, seq_length)
            optimizer.zero_grad()
            output = model(data.float())
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss.update(loss)
            t.set_postfix({'loss': train_loss.avg.item()})
            t.update(1)
    return train_loss.avg.item()
def test(model, device, test_loader, train_loader,verbose=True):
    model.eval()
    input_channels = 1
    seq_length = 140
    test_loss = 0
    correct = 0
    test_acc = Metric('test_acc')
    maximum = 0
    minimum = 0
    for local_batch, local_labels in train_loader:
        if(local_batch.max()>maximum):
            maximum = local_batch.max()
        if(local_batch.min()<minimum):
            minimum = local_batch.min()
    minimum = minimum.numpy()
    maximum = maximum.numpy()
    with torch.no_grad():
        for data, target in test_loader:
            X = data.numpy()
            target = target
            X_std = (X - minimum) / (maximum - minimum)
            X_scaled = X_std * (1 - 0) + 0
            data = torch.from_numpy(X_scaled)
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            data = data.view(-1, input_channels, seq_length)
            output = model(data.float())
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            test_acc.update((pred == target.view_as(pred)).float().mean())
    test_loss /= len(test_loader.dataset)
    print(correct.float()/len(test_loader.dataset))
    return test_acc.avg.item() * 100.

def test_with_integer(model, device, test_loader, train_loader,verbose=True, integer=False):
    model.eval()
    input_channels = 1
    seq_length = 140
    test_loss = 0
    correct = 0
    test_acc = Metric('test_acc')
    maximum = 0
    minimum = 0
    for local_batch, local_labels in train_loader:
        if(local_batch.max()>maximum):
            maximum = local_batch.max()
        if(local_batch.min()<minimum):
            minimum = local_batch.min()
    minimum = minimum.numpy()
    maximum = maximum.numpy()
    with torch.no_grad():
        for data, target in test_loader:
            X = data.numpy()
            target = target
            X_std = (X - minimum) / (maximum - minimum)
            X_scaled = X_std * (1 - 0) + 0
            if integer:      
                X_scaled = X_scaled*255
                X_scaled = np.clip(X_scaled,0,255)
                X_scaled = X_scaled.astype(np.uint8)
                X_scaled = X_scaled.astype(np.float32) 
            data = torch.from_numpy(X_scaled)
            data = data.view(-1, input_channels, seq_length)
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_acc.update((pred == target.view_as(pred)).float().mean())
    test_loss /= len(test_loader.dataset)
    return test_acc.avg.item() * 100.

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
    def update(self, val):
        self.sum += val.cpu()
        self.n += 1
    @property
    def avg(self):
        return self.sum / self.n