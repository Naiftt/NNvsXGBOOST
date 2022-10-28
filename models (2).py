import torch.nn.functional as F
import torch
from torch import nn 
class Net(torch.nn.Module):
    
    def __init__(self, inputsize):
        super().__init__()
        self.fc1 = nn.Linear(inputsize, 10)
        self.fc2 = nn.Linear(10, 1)
#         self.fc2 = nn.Linear(100, 200)
#         self.fc3 = nn.Linear(200, 400)
#         self.fc4 = nn.Linear(400, 100)
#         self.fc5 = nn.Linear(100, 50)
#         self.fc6 = nn.Linear(50, 1)
        self.m = nn.Sigmoid()
        self.m2 = nn.Tanh()

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = self.fc6(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))