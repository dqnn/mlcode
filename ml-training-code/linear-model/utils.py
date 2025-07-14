import torch 
import torchvision
from IPython import display
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

#  train code
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    # [0,0] * 3  
    metric = Accumulator(3)
    # X is the training data [], y is the label [8,3,2]
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    #  [train loss, train accuracy]
    return metric[0]/metric[2], metric[1]/metric[2]

#draw the graph

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend =None, xlim=None, 
                 ylim = None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-', 'r:'), 
                 nrows =1, ncols=1, figsize=(3.5, 2.5)):
        from d2l import torch as d2l
        d2l.use_svg_display()
        # 增量地绘制多条线
        if legend is None:
            legend = []
        
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
    
#  loss is cross_entropy
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    
    
def evaluate_accuracy(net, data_iter):
    # here we just call manual softmax one, not pytorch one
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            # it will return add([1.0, 256])
            # accuracy(net(X), y)--> return sum of values which same as label
            #  y.numel() ---> count of elements in y, return 256
            metric.add(accuracy(net(X), y), y.numel())
    print(metric.data)
    return metric[0]/metric[1]



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
    
    # y_hat.shape is [256, 10], y is [8,2,1,2,4,5...] length = 256
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        #  torch.tensor([[0.1, 0.3,0.6], [0.3,0.2, 0.5]])
        # y_hat.argmax(axis=1) → [2, 2]， real y = [0,2]
        y_hat = y_hat.argmax(axis =1)
    #  reset y_hat data type to be same as y, and compare each element whether they are the same or not
    cmp = y_hat.type(y.dtype) == y
    #  this will return 1
    return float(cmp.type(y.dtype).sum())

# cross_entropy is from shannon's theory, entropy = sum(-p(i) * log(p(i)))
# y_hat.shape=[256,10], y.shape=[256,1], 
# the function will pick the real probabilty from y_hat with index in y(label), then use torch.log to calculate cross entropy 
# y_hat means predictions（0.5）, the return value of softmax, y means real value（0-9）, like label
#  range(len(y_hat)) means how many items(rows), y is label
# so it will return y_hat predictions for each class
# range(len(y_hat)) -> range(row_size_of_y_hat)--> range(256)-> [0,255]
# y_hat[[0,2], [0,2]]--> two elements from 2 rows in y_hat, [0,0] and [1,2]

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])