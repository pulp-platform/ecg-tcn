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
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, prefix, suffix, list_IDs):
        'Initialization'
        self.data = torch.load('data/' + prefix + '_' + suffix + '.pt')
        self.labels = torch.load('data/' + prefix + '_' + suffix + '_labels.pt').view(len(self.data))
        self.list_IDs = list_IDs
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        # Load data and get label
        X = self.data[ID]
        y = self.labels[ID] - 1

        return X, y