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

import os
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nemo
from collections import OrderedDict
from tqdm import tqdm
import random

from utils.util_functions import *
from utils.train_utils import *
from utils.data_loading import *
from utils.model_maker import *
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
export_path = 'model/nemo/' # for both onnx and activations
onnx_name = 'ECG-TCN.onnx'

os.makedirs(export_path, exist_ok=True) 
export_onnx_path = export_path + onnx_name
# Set device
device = "cpu"
cuda_device = "cuda:0"
device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
print('Device: %s' % device)

# pass to device
dummy_input_net = torch.randint(0,256,(1, 1, 140)).type(torch.FloatTensor)
dummy_input_net_float = torch.randn((1, 1, 140))
###########################################################
dummy_input_net = dummy_input_net.to(device)
model = Small_TCN().to(device)

#Split data into training and validation
l = [i for i in range(500)]
params = {'batch_size': 30,
          'shuffle': True,
          'num_workers': 4}
training_set = Dataset('ECG5000', 'train',range(500))
train_loader = torch.utils.data.DataLoader(training_set, **params)
test_set = Dataset('ECG5000', 'test', range(4500))
test_loader = torch.utils.data.DataLoader(test_set, **params)
criterion = nn.CrossEntropyLoss()
#Optimizer
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
input_channels = 1
seq_length = 140
max_epochs = 30
for epoch in range(max_epochs):
    train(model, device, train_loader, optimizer, epoch, verbose=True)

import copy
eps_in = 1.0/255
acc = test(model, device, test_loader, train_loader)
print("\nFullPrecision accuracy: %.02f%%" % acc)

model_nemo = copy.deepcopy(model)
model_nemo = nemo.transform.quantize_pact(model_nemo,remove_dropout=True, dummy_input=torch.randn((1,1,140)).to(device))
precision = model_nemo.export_precision() 
for i, key in enumerate(precision):
    for i, keys in enumerate(precision[key]):
        precision[key][keys] = 16
model_nemo.change_precision(bits=1, min_prec_dict=precision)
acc = test(model_nemo, device, test_loader, train_loader)
print("\nFakeQuantized @ 16b accuracy (first try): %.02f%%" % acc)
with model_nemo.statistics_act():
    _ = test(model_nemo, device, test_loader, train_loader)
model_nemo.reset_alpha_act()
acc = test(model_nemo, device, test_loader, train_loader)
print("\nFakeQuantized @ 16b accuracy (calibrated): %.02f%%" % acc)
for i, key in enumerate(precision):
    for i, keys in enumerate(precision[key]):
        if(keys=='W_bits'):
            precision[key][keys] = 7
        elif(keys=='x_bits'):
            precision[key][keys] = 8
model_nemo.change_precision(bits=1, min_prec_dict=precision)
acc = test(model_nemo, device, test_loader, train_loader)
print("\nFakeQuantized @ 8b accuracy: %.02f%%" % acc)

model_nemo.qd_stage(eps_in=eps_in)

acc = test(model_nemo, device, test_loader, train_loader)
print("\nQuantizedDeployable @ 8b-precision accuracy: %.02f%%" % acc)

model_nemo.id_stage()
acc = test(model_nemo, device, test_loader, train_loader)
print("\nIntegerDeployable @ 8b-precision accuracy: %.02f%%" % acc)
acc = test_with_integer(model_nemo, device, test_loader, train_loader, integer=True)
print("\nIntegerDeployable @ 8b-precision accuracy (for real): %.02f%%" % acc)


print('saving golden activations here: ', export_path)
buf_in, buf_out = get_intermediate_activations(model_nemo, dummy_input_net)
t = dummy_input_net[0].cpu().detach().numpy()
np.savetxt(export_path+'input.txt', t.flatten(), '%.3f', newline=',\\\n', header = 'input (shape %s)' % str(list(t.shape)))
golden_act_names = ['act0', 'act1', 'act2','upsamplerelu', 'reluadd1','act3','act4','reluadd2', 'act5' ,'act6','reluadd3','linear']
L = len(golden_act_names)
for l in range(L):
    t = np.moveaxis(buf_out[golden_act_names[l]][-1].cpu().detach().numpy(), 0, -1)
    np.savetxt(export_path+'out_layer%d.txt' % l, t.flatten(), '%.3f', newline=',\\\n', header = golden_act_names[l] + ' (shape %s)' % str(list(t.shape)))
nemo.utils.export_onnx(export_onnx_path, model_nemo, model_nemo, (1,140), round_params=True)

print("done")




