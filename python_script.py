print('hello world STARTING!')

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import random
import copy
import os
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sin_target_function(Z):
    results = np.zeros((Z.shape[0], 2))
    results[:,0] = Z[:, 0]
    results[:,1] = np.sin(3*Z[:, 0])
    return results

def generate_noise(samples):
    '''
    Generate `samples` samples of uniform noise in 
    ([-1,1], [-1,1])
    '''
    return np.random.uniform(-1, 1, (samples, 2))

def sample_from_target_function(samples, func):
    '''
    sample from the target function
    '''
    Z = generate_noise(samples)
    gaus = noise_mean + noise_sd*torch.randn(Z.shape)
    if func == 'swiss':
        return torch.Tensor(swiss_target_function(Z))+gaus
    elif func == 'circ':
        return torch.Tensor(circ_target_function(Z))+gaus
    elif func == 'sin':
        return torch.Tensor(sin_target_function(Z))+gaus
    
def noisify_data(data, sd = .1):
    return data + sd*torch.randn_like(data)

# For our "True Data"
noise_mean = 0
noise_sd = .05
batch_size = 64
train_data = sample_from_target_function(6400, func = 'sin')
print(train_data.shape, type(train_data))
plt.scatter(train_data[:,0], train_data[:,1])
plt.show()

# Load into data loader for epochal training:
train_loader = torch.utils.data.DataLoader(train_data, shuffle = False, batch_size = batch_size)

class Split_Judge(nn.Module):
    # takes 2 batches of samples. says which batch is better
    def __init__(self, sr_input_dim):
        super(Split_Judge, self).__init__()
        self.fu1 = nn.Linear(sr_input_dim, 128) # each of these should just be normal (d) sized (not r_input_dim)
        self.fl1 = nn.Linear(sr_input_dim, 128)
        
        self.fu2 = nn.Linear(self.fu1.out_features, self.fu1.out_features//2)
        self.fl2 = nn.Linear(self.fl1.out_features, self.fl1.out_features//2)
        combined = self.fu2.out_features+self.fl2.out_features
        
        self.f3 = nn.Linear(combined, combined//2)
        
        self.f4 = nn.Linear(self.f3.out_features, 1) 
        
    # forward method
    def forward(self, upper, lower):
        up1 = F.relu(self.fu1(upper))
        lo1 = F.relu(self.fl1(lower))
        
        up2 = F.relu(self.fu2(up1))
        lo2 = F.relu(self.fl2(lo1))
        
        #print(up2, lo2)
        out = torch.cat((up2, lo2), -1) # changed from 1
        #print(out.shape)
        out = F.relu(self.f3(out))
        return torch.sigmoid(self.f4(out))
    
    
def plot_boundary():
    # Sample data space, then find each sample's nearest manifold element:
    manif_picks = torch.Tensor(sample_from_target_function(40, 'sin'))
    data_picks = torch.Tensor(generate_noise(1000))

    dist_mat = pairwise_distances(manif_picks, data_picks)
    #print(dist_mat, dist_mat.shape)
    near_inds = torch.argmin(dist_mat, axis=0)
    #print(near_inds.shape) # for each datapoint, what is its closest manif point?

    ind = 0
    J.cpu()
    J_responses = []
    for row in data_picks:
        manif_point = manif_picks[near_inds[ind]]
        #print(row, manif_point)
        row = row@trans_to
        manif_point = manif_point@trans_to
        J_out = J(row, manif_point)
        #print(J_out)
        J_responses.append(J_out.data.item())    

    plt.scatter(data_picks[:,0], data_picks[:,1], c = J_responses, alpha = .5)
    plt.scatter(manif_picks[:,0], manif_picks[:,1])
    plt.colorbar()
    plt.title('yellow: judge prefers manif point. purp: judge prefers data point')
    plt.plot()
    
def plot_lodim_boundary():
    # Sample data space, then find each sample's nearest manifold element:
    manif_picks = torch.Tensor(sample_from_target_function(40, 'sin'))
    data_picks = torch.Tensor(generate_noise(1000))

    dist_mat = pairwise_distances(manif_picks, data_picks)
    #print(dist_mat, dist_mat.shape)
    near_inds = torch.argmin(dist_mat, axis=0)
    #print(near_inds.shape) # for each datapoint, what is its closest manif point?

    ind = 0
    J.cpu()
    J_responses = []
    for row in data_picks:
        manif_point = manif_picks[near_inds[ind]]
        #print(row, manif_point)
        J_out = J(row, manif_point)
        #print(J_out)
        J_responses.append(J_out.data.item())    

    plt.scatter(data_picks[:,0], data_picks[:,1], c = J_responses, alpha = .5)
    plt.scatter(manif_picks[:,0], manif_picks[:,1])
    plt.colorbar()
    plt.title('yellow: judge prefers manif point. purp: judge prefers data point')
    plt.plot()
    
def pairwise_distances(x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y = x
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist
        
        
J = Split_Judge(train_data.shape[1]).to(device)
J_optimizer = optim.Adam(J.parameters(), lr = .0002)
criterion = nn.BCELoss()

hi_noise_sd = .4

x = next(iter(train_loader))
hi_noised = noisify_data(x, hi_noise_sd)
plt.scatter(hi_noised[:,0], hi_noised[:,1], label = 'high-noised')
plt.scatter(x[:,0], x[:,1], label = 'true')
plt.legend(loc = 'best')
plt.show()

n_epoch = 50
# "symmetric training"
loss_list = []
acc_list = []
for epoch in range(1, n_epoch+1):
    J_losses = []
    J_accs = []
    for batch_idx, x in enumerate(train_loader):
        hi_noised = noisify_data(x, hi_noise_sd)
        x = x.to(device)
        hi_noised = hi_noised.to(device)
        
        J_0 = J(x, hi_noised) # should return 0
        #print(J_0)
        J_1 = J(hi_noised, x) # should return 1
        #print(J_1)
        
        J_loss = criterion(J_0, torch.zeros_like(J_0)) + criterion(J_1, torch.ones_like(J_1))
        
        J_pred = [1 if x>.5 else 0 for x in J_0] # should predict all 0 here
        J_acc = 1 - sum(J_pred)/len(J_pred) # 1 - error_rate
        #print(J_loss)
        J_loss.backward()
        J_optimizer.step()
        
        #print(J_loss.item())
        J_losses.append(J_loss.item())
        J_accs.append(J_acc)
    
    loss_list.append(torch.mean(torch.FloatTensor(J_losses)))
    acc_list.append(torch.mean(torch.FloatTensor(J_accs)))

plt.plot(range(n_epoch), loss_list, label = 'losses')
plt.plot(range(n_epoch), acc_list, label = 'accuracy')
plt.hlines(1, 0, n_epoch)
plt.xlabel('Training Time')
plt.legend(loc = 'best')
#plt.ylabel('J loss')
plt.title('J thru time')
plt.show()

print('hello world ENDING')
