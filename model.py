import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=0)

        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 10)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm1d(128)

        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.flatten(1)
        # x = self.dropout2(x)

        x = self.fc1(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output
