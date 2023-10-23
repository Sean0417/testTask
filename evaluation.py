import torch
import torch.nn as nn

class Evaluation():
    def __init__(self,model,learningRate,) -> None:
            
        self.criterion = torch.nn.MSELoss(size_average=False)

        self.optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)