import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import pandas as pd
import random

class VisualEmbedding(nn.Module):
    def __init__(self, out_size = 512):
        super(VisualEmbedding, self).__init__()
        self.conv = torchvision.models.googlenet(pretrained=True)
        self.conv = nn.Sequential(*list(self.conv.children())[:-2])
        self.linear = nn.Linear(out_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear(self.relu(self.conv(x).view(x.shape[0], -1))) # (batch_size, emb_size)

        return x