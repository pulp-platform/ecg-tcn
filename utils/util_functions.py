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
import nemo
from collections import OrderedDict

def print_summary(model):
    summary = nemo.utils.get_summary(model, tuple(torch.squeeze(dummy_input_net, 0).size()), verbose=True)
    print(summary[1])

def get_intermediate_activations(net, dummy_input_net):
    l = len(list(net.named_modules()))
    buffer_in  = OrderedDict([])
    buffer_out = OrderedDict([])
    hooks = OrderedDict([])
    def get_hk(n):
        def hk(module, input, output):
            buffer_in  [n] = input
            buffer_out [n] = output
        return hk
    for i,(n,l) in enumerate(net.named_modules()):
        hk = get_hk(n)
        hooks[n] = l.register_forward_hook(hk)

    outputs = net(dummy_input_net)
    return buffer_in, buffer_out