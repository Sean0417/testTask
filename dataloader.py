import numpy as np
import torch
from torch.utils.data import Dataset # Dataset 是一个抽象类，不能实例化，只能被继承
from torch.utils.data import DataLoader # DataLoader is a class to help us loading data in Pytorch
import pandas as pd
import matplotlib.pyplot as plt
def readCSV(filepath):
        odata = pd.read_csv(filepath,usecols=[8])
        odata = np.array(odata)
        odata_len = odata.shape[0]
        x_data = []
        y_data = []
        for i in range(odata_len-6):
            _x = odata[i:(i+6),0]
            x_data.append(_x)
            # _y = odata[i+6]
            y_data.append(odata[i+6,0])
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data, y_data


class OzoneDataset(Dataset):
    def __init__(self,x_data,y_data) -> None:
 
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)

    def __len__(self):
        return self.x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

def train_data_prepare(filepath):
    x_data, y_data = readCSV(filepath=filepath)
    train_size = int(len(x_data)*0.7)
    train_x = x_data[:train_size]
    train_y = y_data[:train_size]
    train_dataset = OzoneDataset(train_x,train_y)
    train_loader = DataLoader(dataset=train_dataset,batch_size=40,shuffle=True,num_workers=2)
    return train_loader

def test_data_prepare(filepath):
    x_data, y_data = readCSV(filepath=filepath)
    test_size = int(len(x_data)*0.7)
    test_x = x_data[test_size:]
    test_y = y_data[test_size:]
    test_dataset = OzoneDataset(test_x,test_y)
    test_loader = DataLoader(dataset=test_dataset,batch_size=40,shuffle=False,num_workers=2)
    return test_loader

def plot(filepath):
    x_data, y_data = readCSV(filepath=filepath)
    data_close = x_data.astype('float32') # 转换数据类型
    plt.plot(data_close)
    plt.xticks(np.arange(0, 20000, step = 5000))
    plt.savefig('data.png', format = 'png', dpi = 300)
    plt.close()

def get_real_y(filepath): # 返回值为numpy
        odata = pd.read_csv(filepath,usecols=[8])
        odata = np.array(odata)
        # fodata = []
        # for x in odata:
        #      fodata.append(x[0])
        # return fodata
        return odata


# # 
# train_x, train_y = readCSV('data/data.csv')
# train_dataset = OzoneDataset(train_x,train_y)
# print(train_dataset.__len__())

# y = get_real_y("data/data.csv")
# print(y)