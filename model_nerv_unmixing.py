# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:20:33 2023

@author: 19652
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as scio
    
class CustomDataSetuniform(Dataset):
    def __init__(self, X, frame_idx, transform, vid_list=[None], frame_gap=1,visualize=False):
        #self.main_dir = main_dir
        self.transform = transform
        #frame_idx=[]
        accum_img_num = []

        rows, columns = X.shape
        col=int(np.sqrt(columns))
        self.all_imgs=X.reshape(rows,col,col)
        num_frame = rows
       # for img_id in range(0,156):
       #     frame_idx.append(num_frame)
       #  num_frame += 1   

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        accum_img_num.append(num_frame)
        self.frame_idx = [float(x) / 156 for x in frame_idx]
        self.accum_img_num = np.asfarray(accum_img_num)
        if None not in vid_list:
           self.frame_idx = [self.frame_idx[i] for i in vid_list]
        #frame_gap = np.diff(frame_idx, prepend=frame_idx[0])   
        self.frame_gap = frame_gap

    def __len__(self):
        return len(self.frame_idx) // self.frame_gap

    def __getitem__(self, idx):
        valid_idx = idx * self.frame_gap 
        image =self. all_imgs[valid_idx,:,:]
        tensor_image = self.transform(image)
        #tensor_image = torch.tensor(image)
        frame_idx = torch.tensor(self.frame_idx[valid_idx])
        return tensor_image, frame_idx     
    
    


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)

def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer

def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer
def MLP1(dim_list, act, bias=True): 
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)

class Generator(nn.Module):
    def __init__(self, **kargs):
       super().__init__()
       stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
       self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
       mlp_dim_list = [kargs['embed_length']] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]
       self.stem = MLP1(dim_list=mlp_dim_list, act=kargs['act'])
       #self.batch_norm = nn.BatchNorm1d(self.fc_h * self.fc_w * self.fc_dim)

    def forward(self, input):
        output = self.stem(input)
       # output = self.batch_norm(output) 
        output = output.view(output.size(0), self.fc_dim, self.fc_h*self.fc_w) 
        return  output
class CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()
        ngf, new_ngf = kargs['ngf'], kargs['new_ngf']
        self.conv_type = 'conv'
        self.conv = nn.Conv2d(ngf, new_ngf, 3, 1, 1, bias=kargs['bias'])
        self.max_pooling = nn.AvgPool2d(kernel_size=3)

    def forward(self, x):
        out = self.conv(x)
        return self.max_pooling(out)


def MLP(dim_list, act, bias=True):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)


class Generator1(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        mlp_dim_list = [kargs['embed_length']] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]
        self.stem = MLP(dim_list=mlp_dim_list, act=kargs['act'])

        
      
        new_ngf =1
        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1,
                bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], conv_type=kargs['conv_type']))
        ngf = new_ngf
        head_layer = nn.Conv2d(ngf, 1, 3, 1, 1, bias=kargs['bias'])
        self.head_layers.append(head_layer)
        self.sigmoid =kargs['sigmoid']

    def forward(self, input):
        output = self.stem(input)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)

        out_list = []
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output) 
            if head_layer is not None:
                img_out = head_layer(output)
                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)
        out_list=torch.stack(out_list)
        out_list=out_list.view(1,1,3)
        return  out_list        
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=5):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        self.init_weights()
       # self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
                                              
 #               self.linear.weight.square_()      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
                #self.linear.weight.square_() 
    def swish(self, x):
        return x * torch.sigmoid(x)    
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
        #return self.swish(self.omega_0 * self.linear(input))
        #return self.leaky_relu(self.omega_0 * self.linear(input))
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
        #return self.swish(intermediate),intermediate
        #return self.leaky_relu(intermediate),intermediate
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=5, hidden_omega_0=5.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        self.out_features=out_features

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6/ hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                #final_linear.weight.square_() 
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, input):
        output = self.net(input)
        output = output.view(output.size(0), 1, self.out_features)                                               
        return  torch.sigmoid(output)

def endab(endmember,abundance,endnum,row):
        out_list = []
        Endmember1=endmember.view(1,endnum)                                                    ##### 
        output= torch.mm(Endmember1,abundance)   
        output= output.view(1,1,row,row) 
        out_list.append(output)   
        return output

    