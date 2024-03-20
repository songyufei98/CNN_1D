import torch.nn as nn


class CNN_1D(nn.Module):
    def __init__(self, in_chanel = 1):
        super(CNN_1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_chanel, 20, 3, 1, 0),  #out_channel = 20, kernel_size = 3, stride = 1
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.ReLU()
            )
        self.fc1 = nn.Linear(50 * 2, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        out = self.fc2(x)
        return out
