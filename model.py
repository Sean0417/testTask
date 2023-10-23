import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset # Dataset 是一个抽象类，不能实例化，只能被继承
from torch.utils.data import DataLoader # DataLoader is a class to help us loading data in Pytorch

# A = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7],[3,4,5,6,7,8]]).view(-1,1,6)
# # print(A)
# B = torch.from_numpy(np.array([[7],[8],[9]]))
class LSTM_Regression(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1, num_layers = 1):
        super(LSTM_Regression,self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x,_ = self.lstm(_x) #(x是输出维度为 seq*batch*hidden_size)
        s,b,h = x.shape
        x = x.view(s*b,h)
        x = self.fc(x)
        x.view(s,b,-1)
        return x

# model = LSTM_Regression(input_size=6,hidden_size=1) # 输入一定要是tensor， 且维度为 seq * batch* input_size
# y_pred = model(A)
# print(y_pred)
