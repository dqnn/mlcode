import torch 
import torchvision
from IPython import display
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from animator_utils import Animator
from accumulator_utils import Accumulator

    
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