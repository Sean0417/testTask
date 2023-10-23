from model import LSTM_Regression
import torch.nn
import torch
import dataloader
from dataloader import train_data_prepare
from dataloader import test_data_prepare
from dataloader import plot
import matplotlib.pyplot as plt
import numpy as np
import time
def main():
    loss = []
    # 1. prepare dataset
    precon = 6
    filepath = "data/data.csv"
    hidden = 6

    train_loader = train_data_prepare(filepath=filepath) # train_loader 里面都是Tensor
    test_loader = test_data_prepare(filepath=filepath)
    plot(filepath=filepath)


    t0 = time.time()
    model = LSTM_Regression(precon,hidden_size=hidden)
    
    criterion = torch.nn.MSELoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

    for epoch in range(1000):
        epoch_loss = []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = torch.tensor(labels,dtype=torch.float32).view(-1,1)
            inputs = torch.tensor(inputs, dtype=torch.float32).view(-1,1,6)
            # print("labels: ", labels)
            # print("inputs: ", inputs)
            y_pred = model(inputs)
            train_loss = criterion(y_pred, labels) # 此处算出的是每一个batch的loss，是一个int值
            epoch_loss.append(train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step() # 更新参数
            if (i+1) % 1 ==0:
                print('Epoch: {},Batch: {} Loss:{:.5f}'.format((epoch+1),(i+1),train_loss.item()))
            if train_loss < 7:
                break
        # print(epoch_loss)
        loss += epoch_loss #loss是一个list
        if np.min(epoch_loss) < 7:
            break
    

    # 画loss曲线
    plt.figure()
    plt.plot(loss, 'b', label= 'loss')
    plt.title("Train_Loss_curve")
    plt.ylabel("train_loss")
    plt.xlabel("iteration_num")
    plt.savefig('loss.png',format='png',dpi= 200)
    plt.show()
    plt.close

    # 保存模型里的参数
    torch.save(model.state_dict(), 'model_params.pkl')
    t1 = time.time()
    T = t1-t0
    print(
        "The training took %.2f"%(T/60)+"mins."
    )

    tt0 = time.asctime(time.localtime(t0))
    tt1 = time.asctime(time.localtime(t1))
    print("The starting time was ",tt0)
    print("The finishing time was ",tt1)

    # for test
    model = model.eval()
    model.load_state_dict(torch.load("model_params.pkl"))
    y_pred = []
    y_real = dataloader.get_real_y('data/data.csv')
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = torch.tensor(inputs, dtype=torch.float32).view(-1,1,6)
        y_out = model(inputs)
        y_out = y_out.view(-1).data.numpy()
        y_pred.extend(y_out)
    
    test_range = np.array(range(int(0.7*(len(y_real))),int(0.7*(len(y_real)))+int(len(y_pred))))
    plt.plot(y_real,'b', label='real')
    plt.plot(test_range,y_pred, 'r', label='prediction')
    
    plt.plot((int(0.7*(len(y_real))),int(0.7*len(y_real))),(0,100), 'g--')# 分割线
    plt.legend(loc = 'best')
    plt.savefig('result.png',format = 'png', dpi = 200)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()