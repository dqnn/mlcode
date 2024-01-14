import torch
import random
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0,0.1, y.shape)
    return x, y.reshape((-1, 1))



def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indexes = list(range(num_examples))
    random.shuffle(indexes)
    for i in range(0, num_examples, batch_size):
        batch_indexes = torch.tensor(indexes[i:min(i+batch_size, num_examples)])
        yield features[batch_indexes], labels[batch_indexes]


def linreg(x, w, b):
    return torch.matmul(x, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2/2

def sgd(params, lr, batch_sie):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad/batch_size
            param.grad.zero_()

    

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 10000)
batch_size = 50

w = torch.normal(0,0.01, (2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

## start to train
lr = 0.01
num_epochs = 40
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')






