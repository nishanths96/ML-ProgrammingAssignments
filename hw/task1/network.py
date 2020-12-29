import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(param):
    # NOTE: Not for Vanilla Classifier
    raise NotImplementedError


def zero_init(param):
    # NOTE: Not for Vanilla Classifier
    raise NotImplementedError


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO: Define the model architecture here
        # Build a feed-forward network
        self.layer1 = nn.Linear(784, 100)
        self.layer2 = nn.Linear(100, 10)
        self.drop_layer = nn.Dropout(p=0.3)
        # NOTE: Not for Vanilla Classsifier
        # TODO: Initalize weights by calling the
        # init_weights method
        self.init_weights('zero')

    def init_weights(self, type):
        # NOTE: Not for Vanilla Classsifier
        if type == 'xavier':
            nn.init.xavier_normal_(self.layer1.weight.data)
            nn.init.xavier_normal_(self.layer2.weight.data)
            nn.init.zeros_(self.layer1.bias.data)
            nn.init.zeros_(self.layer2.bias.data)
        else:
            nn.init.zeros_(self.layer1.weight.data)
            nn.init.zeros_(self.layer2.weight.data)
            nn.init.zeros_(self.layer1.bias.data)
            nn.init.zeros_(self.layer2.bias.data)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.drop_layer(x)
        x = F.log_softmax(self.layer2(x), dim=1)
        return x

    def save(self, ckpt_path):
        torch.save(self, ckpt_path)

    def load(self, ckpt_path):
        model = torch.load(ckpt_path)