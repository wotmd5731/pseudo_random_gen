# -*- coding: utf-8 -*-

import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import os
import csv
"""
NOT USED.

leaky_relu사용.
Pred 값이 - ~ +가 나옴
real target value is only 0 or 1 value


"""

        
class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(45 ,512 , 3,batch_first = True)
        self.fc = nn.Linear(512 ,45)
        
        
        
    def forward(self, x,hx,cx):
        out , (hx, cx) = self.lstm(x, (hx, cx))
        x = F.leaky_relu(self.fc(out))
#        x = F.sigmoid(self.fc(out))
        
        return x ,hx,cx
    
cx = Variable(torch.zeros(3,1, 512))
hx = Variable(torch.zeros(3,1, 512))


f = open('data.csv','r',encoding='utf-8')
rdr = csv.reader(f)
data = []
for line in rdr:
    data.append(line[-7:])
#    print(line[-7:1])
f.close()
data = data[3:]
np_data = np.array(data, dtype=np.long)
torch_data = torch.from_numpy(np_data).type(torch.LongTensor)


main_num = torch_data[:,:6]
bonus_num = torch_data[:,6].unsqueeze(1)

#flip data seq
inv_idx = torch.arange(main_num.size(0)-1, -1, -1).long()
main_num = main_num.index_select(0, inv_idx)
bonus_num = bonus_num.index_select(0, inv_idx)


main_data = Variable(torch.zeros(main_num.size(0),46).scatter_(1,main_num,1)[:,1:].unsqueeze(0))
bonus_data = Variable(torch.zeros(bonus_num.size(0),46).scatter_(1,bonus_num,1)[:,1:].unsqueeze(0))


net = network()
#loss = nn.CrossEntropyLoss()
#crit = nn.KLDivLoss()
crit = nn.MSELoss()
#crit = nn.BCELoss(size_average = True)
opti = optim.Adam(net.parameters(),lr=0.0001)

for i in range(main_data.size(1)):
    out,hx,cx = net(main_data[0,i,:].view(1,1,-1),hx,cx)
    out = out.view(-1)
    if i == main_data.size(1)-1 :
        print(out.data.numpy())
        break;
    target = main_data[0,i+1,:]
    loss = crit(out, target)
    print(out.data.numpy())
    print('i : ', i ,' loss :',loss.data[0])
    net.zero_grad()
    loss.backward(retain_graph=True)
    nn.utils.clip_grad_norm(net.parameters(), 10)  # Clip gradients (normalising by max value of gradient L2 norm)
    opti.step()
        





#out ,hx,cx = net(main_data,hx,cx)
#last_out = out[:,-1,:]
#
#
#
#
#
#
#loss = nn.CrossEntropyLoss()
#
#target = Variable(torch.LongTensor(batch_size).random_(0, classes_no-1))
#
#err = loss(last_output, target)
#err.backward()
#
#
#
#
