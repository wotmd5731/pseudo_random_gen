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
import matplotlib.pyplot as plt
"""
main1~3.py 는 NOT USED..

%matplotlib inline

leaky_relu사용.-> sigmoid 로 변경.
Pred 값이 - ~ +가 나옴 -> pred 값이 0~1 로 나옴.
Binary Cross Entropy , KLD 사용 가능. 현재 BCE 사용 중.
real target value is only 0 or 1 value.
one to one lstm model -> many to one lstm model 변경
one to one (seq_len = 1) 사용시 step_init = False 로 만들어야됨.
step_init True 일경우 과거 데이터 없음.

random batch 적용.
마지막 이터레이션 에서 이번 값에 대한 Pred 출력함.
출력 에 +1 해야 실제 추첨 번호가 됨.
-> 동일한 batch 입력해서 mean 수행하는 식으로 진행.

"""

num_layer = 4
num_hidden = 512



class network(nn.Module):
    def __init__(self):
        global num_layer,num_hidden
        super().__init__()
        self.lstm = nn.LSTM(45 ,num_hidden , num_layer,batch_first = True)
        self.fc = nn.Linear(num_hidden ,45)
        
        
        
    def forward(self, x,hx,cx):
        out , (hx, cx) = self.lstm(x, (hx, cx))
#        x = F.leaky_relu(self.fc(out))
        x = F.sigmoid(self.fc(out))
        
        return x ,hx,cx

net = network()
#loss = nn.CrossEntropyLoss()
#crit = nn.KLDivLoss()
#crit = nn.MSELoss()
crit = nn.BCELoss()
opti = optim.Adam(net.parameters(),lr=0.001)

learn_iterate = 100
batch_size = 16
seq_len = 32
#seq_len = 1
#step_init = False #seq_len == 1 then step_init -> false 
step_init = True


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

#test reduce data size 
#main_data=main_data[:,:50,:]



""" 
random batch n_seq 학습 모델. 
학습 횟수 k 
random sampling 한 데이터 batch 개를 통해 n_seq 만큼 학습 하고 값을 pred 함.
"""
plot_loss = []
cx = Variable(torch.zeros(num_layer,batch_size, num_hidden))
hx = Variable(torch.zeros(num_layer,batch_size, num_hidden))

final = False
for i in range(learn_iterate):
    if(i == learn_iterate-1):
        final = True
        
    if step_init:
        cx = Variable(torch.zeros(num_layer,batch_size, num_hidden))
        hx = Variable(torch.zeros(num_layer,batch_size, num_hidden))
    
    random.seed()
#    sample_num = [random.randint(0,main_data.size(1)-seq_len+1) for b in range(batch_size)]
    """
    main_size - seq_len  이 맞는데 학습 과정에선 시퀀스 +1 값을 통해서 loss를 구함으로  main_size - seq_len -1 으로 변경.  
    if seq_len = 1  then max -> 791 - 1 = 790 이 되고 이때  pred 이후 target 790+1 =791이 존재하지 않음.
    따라서 randint 는 main - seq - 1 이 맞음.
    참고로 randint(0,4) 이면 0~4 가 나온느거임...
    """
    sample_num = [random.randint(0,main_data.size(1)-seq_len - 1) for b in range(batch_size)]
    if final:
        fnum = main_data.size(1)-seq_len
        sample_num = [fnum for b in range(batch_size)]
        print('fnum : ',fnum)
        
    batch_main_data =[]
    for num in sample_num:
        batch_main_data.append(main_data[:,num:num+seq_len,:])
#        print('sampled data size ' , main_data[:,num:num+seq_len,:].size())
    
    batch_main_data = torch.cat(batch_main_data,0)
    out,hx,cx = net(batch_main_data,hx,cx)
    print('smple ',sample_num,'seq_len',seq_len)
    
    last_out = out[:,-1,:].mean(0)
    
    choose_num = last_out.topk(6)[1].sort()[0].data.numpy()
    print('choose ', choose_num)
    
#    if i ==  :
#        print('finally prediction out :')
#        print(last_out.data.numpy())
#        break;
    
    if final:
        break
    
    batch_target_data =[]
    for num in sample_num:
        batch_target_data.append(main_data[:,num+seq_len,:])
    batch_target_data = torch.cat(batch_target_data,0).mean(0)
    
    
    
    print('target ',i)
    loss = crit(last_out, batch_target_data)
    
#    print(last_out.data.numpy())
    print('i : ', i ,' loss :',loss.data[0] )
    plot_loss.append(loss.data.numpy())
#    plt.clf()
    plt.plot(plot_loss)
    plt.draw()
    plt.pause(0.01)
    print('========end========')
    net.zero_grad()
    loss.backward(retain_graph=True)
    nn.utils.clip_grad_norm(net.parameters(), 10)  # Clip gradients (normalising by max value of gradient L2 norm)
    opti.step()
        






