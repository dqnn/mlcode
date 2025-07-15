from IPython import display
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

#  data[0] = 存储正确预测的数量 
#  data[1] = 预测的总数量
class Accumulator:
    def __init__(self, n):
        
        # [0,0] * n --> repeat [0,0] n times, for example
        # [0,0] * 2 -> [0,0,0,0]
        self.data = [0,0] * n
    
    def add(self, *args):
        #  will return [存储正确预测的数量, 预测的总数量]
        # 如果 self.data = [0,0,0,0]，args = [1,0],[256]，那么：
        # zip(self.data, args) ➜ [(1.0, 4), (2.0, 5), (3.0, 6)]
        ## print('agrs: ' + str(args)) --> agrs: (212.0, 256)
        self.data =[a+float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0]* len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]