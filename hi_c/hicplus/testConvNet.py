# Author: Yan Zhang
# Email: zhangyan.cse (@) gmail.com

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gzip
from model import Net
import model
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import sys
import torch.nn as nn
import scipy.ndimage

use_gpu = 1

conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5


down_sample_ratio = 16
epochs = 10
HiC_max_value = 100



# This block is the actual training data used in the training. The training data is too large to put on Github, so only toy data is used.
# cell = "GM12878_replicate"
# chrN_range1 = '1_8'
# chrN_range = '1_8'

# low_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/'+cell+'down16_chr'+chrN_range+'.npy.gz', "r")).astype(np.float32) * down_sample_ratio
# high_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/original10k/'+cell+'_original_chr'+chrN_range+'.npy.gz', "r")).astype(np.float32)

# low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
# high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)


low_resolution_samples = np.load(gzip.GzipFile('../data/GM12878_replicate_down16_chr17_17.npy.gz', "r")).astype(np.float32) * down_sample_ratio
#low_resolution_samples = np.load(gzip.GzipFile('../data/GM12878_replicate_down16_chr19_22.npy.gz', "r")).astype(np.float32) * down_sample_ratio

low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)

batch_size = low_resolution_samples.shape[0]

# Reshape the high-quality Hi-C sample as the target value of the training.
sample_size = low_resolution_samples.shape[-1]
padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
half_padding = padding / 2
output_length = sample_size - padding


print(low_resolution_samples.shape)

lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)

production = False
try:
    high_resolution_samples = np.load(gzip.GzipFile('../data/GM12878_replicate_original_chr19_22.npy.gz', "r")).astype(np.float32)
    high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)
    Y = []
    for i in range(high_resolution_samples.shape[0]):
        no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        Y.append(no_padding_sample)
    Y = np.array(Y).astype(np.float32)
    hires_set = data.TensorDataset(torch.from_numpy(Y), torch.from_numpy(np.zeros(Y.shape[0])))
    hires_loader = torch.utils.data.DataLoader(hires_set, batch_size=batch_size, shuffle=False)
except:
    production = True
    hires_loader = lowres_loader

Net = Net(40,40)
Net.load_state_dict(torch.load('../model/test_epoch900.pth'))
#if use_gpu:
    #Net = Net.cuda()

#%% plot the low res matrix
cmap = plt.get_cmap('Reds')

#plt.imshow(arr, interpolation='none', cmap=cmap, vmax=0.5)
plt.imshow(lowres_set[2710][0][0,:,:], interpolation='none', cmap=cmap, vmax=80)

#%% plot the prediction
#print(lowres_set[0][0])
x=torch.unsqueeze(lowres_set[2710][0], 0)
#x=x.to(torch.device('cuda'))
pred=Net(x)
plt.imshow(pred[0,0,:,:].detach().numpy(),interpolation='none', cmap=cmap, vmax=35)

#%%

# %%
