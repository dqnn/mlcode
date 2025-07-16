import torch
import random
from d2l import torch as d2l



### python ml-training-code/ch3_linear_model/linear_code.py

# genearte label and training data
def synthetic_data(w, b, num_examples):
    #generate label data
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = x@w + b # same as y = torch.matmul(x, w) + b
    ## genearte training data
    y += torch.normal(0,0.01, y.shape)
    # only 1 column
    return x, y.reshape((-1, 1))


##generate mini_batch data for training
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indexes = list(range(num_examples))
    random.shuffle(indexes)
    for i in range(0, num_examples, batch_size):
        batch_indexes = torch.tensor(indexes[i:min(i+batch_size, num_examples)])
        yield features[batch_indexes], labels[batch_indexes]


def linreg(x, w, b):
    return x@w + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2/2

#genearte grad descent
#params = [w, b]
def sgd(params, lr, batch_sie):
    ## here we shutdown all grad descent computation, update [w, b]
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad/batch_size
            param.grad.zero_()

    


## start to train, lr = learning rate
lr = 0.005
num_epochs = 30
net = linreg
loss = squared_loss

w = torch.normal(0,0.01, (2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)
        ## we get sum of training data results, 
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    # here we print result after one epoch, 
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')






