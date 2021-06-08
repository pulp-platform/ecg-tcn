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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding1D, Conv1D, Add, Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Dense, Reshape, ZeroPadding2D, Conv2D
import tensorflow as tf

def tf_model(X_train_1D):
    
    input_dim = 1
    kernel_s = 11
    layers = 3
    filt = 11
    drop_r = 0.3
    classes = 5
    
    ip = Input(shape=(X_train_1D[0].shape))
    x = Conv1D(2,kernel_size=11,dilation_rate=1,activation='relu',padding='same', use_bias=False)(ip)
    x = BatchNormalization()(x)
    x1 = ZeroPadding1D(padding=(0,0))(x)
    x1 = Conv1D(filt,kernel_size=kernel_s,dilation_rate=1,activation='relu', padding = 'causal',kernel_initializer='he_uniform', use_bias=False)(x1)
    x1 = Dropout(drop_r)(x1)
    x1 = ZeroPadding1D(padding=(0,0))(x1)
    x1 = Conv1D(filt,kernel_size=kernel_s,dilation_rate=1,activation='relu', padding = 'causal',kernel_initializer='he_uniform', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(drop_r)(x1)
    conv = Conv1D(filt,kernel_size=1,padding='same', use_bias=False)(x)
    conv = BatchNormalization()(conv)
    added_1 = Add()([x1, conv])
    out = Activation('relu')(added_1)



    for i in range(layers-1):
        x = ZeroPadding1D(padding=(0,0))(out)
        x = Conv1D(filt,kernel_size=kernel_s,dilation_rate=2**(i+1),activation='relu', padding = 'causal',kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(drop_r)(x)
        x = ZeroPadding1D(padding=(0,0))(x)
        x = Conv1D(filt,kernel_size=kernel_s,dilation_rate=2**(i+1),activation='relu', padding = 'causal',kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(drop_r)(x)

        added = Add()([x, out])
        out = Activation('relu')(added)
    out = Flatten()(out)
    ep = Dense(classes, use_bias=False)(out)
    ep = Activation('softmax')(ep)

    model_1D = Model(ip,ep)
    return model_1D

def tf_model_zero(X_train):
    batch_size = 30
    input_dim = 1
    kernel_s = 11
    layers = 3
    filt = 11
    drop_r = 0.3
    classes = 5
    epochs = 10
    ip = Input(shape=(X_train[0].shape))
    x = Conv2D(2,kernel_size=(1,11),dilation_rate=1,activation='relu',padding='same', use_bias=False)(ip)
    x = BatchNormalization(fused=True)(x)
    x1 = ZeroPadding2D(padding=((0,0),(1*(kernel_s-1),0)))(x)
    x1 = Conv2D(filt,kernel_size=(1,(1*(kernel_s-1)+1)),dilation_rate=1,activation='relu', padding = 'valid',kernel_initializer='he_uniform', use_bias=False)(x1)
    x1 = Dropout(drop_r)(x1)
    x1 = ZeroPadding2D(padding=((0,0),(1*(kernel_s-1),0)))(x1)
    x1 = Conv2D(filt,kernel_size=(1,(1*(kernel_s-1)+1)),dilation_rate=1,activation='relu', padding = 'valid',kernel_initializer='he_uniform', use_bias=False)(x1)
    x1 = BatchNormalization(fused=True)(x1)
    x1 = Dropout(drop_r)(x1)
    conv = Conv2D(filt,kernel_size=1,padding='same',use_bias=False)(x)
    conv = BatchNormalization(fused=True)(conv)
    added_1 = Add()([x1, conv])
    out = Activation('relu')(added_1)



    for i in range(layers-1):
        x = ZeroPadding2D(padding=((0,0),(2**(i+1)*(kernel_s-1),0)))(out)
        x = Conv2D(filt,kernel_size=(1,(2**(i+1)*(kernel_s-1)+1)),dilation_rate=1,activation='relu', padding = 'valid',kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(fused=True)(x)
        x = Dropout(drop_r)(x)
        x = ZeroPadding2D(padding=((0,0),(2**(i+1)*(kernel_s-1),0)))(x)
        x = Conv2D(filt,kernel_size=(1,(2**(i+1)*(kernel_s-1)+1)),dilation_rate=1,activation='relu', padding = 'valid',kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(fused=True)(x)
        x = Dropout(drop_r)(x)

        added = Add()([x, out])
        out = Activation('relu')(added)
    out = Flatten()(out)
    ep = Dense(classes, use_bias=False)(out)
    ep = Activation('softmax')(ep)

    model = Model(ip,ep)
    return model

def model_weight_setter(model_2D,model_1D):
    for ind, layer in enumerate(model_2D.layers):
        if('conv2d' in layer.name):
            dilation_rate = model_1D.layers[ind].dilation_rate[0]
            if(dilation_rate > 1):
                d = layer.get_weights()
                c = model_1D.layers[ind].get_weights()[0]
                w_ind = 0
                for i,p in enumerate(c):
                    d[0][0,w_ind] = c[i]
                    w_ind = w_ind + 1
                    if(i!=len(c[0])-1):
                        for fill in range(dilation_rate-1):
                            d[0][0,w_ind].fill(0)
                            w_ind = w_ind + 1
                model_2D.layers[ind].set_weights(d)
            else:
                d = layer.get_weights()
                c = model_1D.layers[ind].get_weights()[0]
                d[0][0] = c
                model_2D.layers[ind].set_weights(d)
        elif('batch_normalization' in layer.name):
            d = layer.get_weights()
            c = model_1D.layers[ind].get_weights()
            for b in range(len(d)):
                d[b] = c[b]
            model_2D.layers[ind].set_weights(d)
        elif('dense' in layer.name):
            d = layer.get_weights()
            c = model_1D.layers[ind].get_weights()
            for b in range(len(d)):
                d[b] = c[b]
            model_2D.layers[ind].set_weights(d)