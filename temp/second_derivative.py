"""
Second derivative using Hessian matrix

Training on MNist with first five classes

"""


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from tensorflow import keras
from torch.utils.data import Dataset
import torch.nn.functional as F

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel




class MNISTDataset(Dataset):
    def __init__(self, image, label):
        self.image = torch.tensor(image, dtype = torch.float32)
        self.label = label

    def __len__(self): return len(self.label)

    def __getitem__(self, idx):
        image_ = self.image[idx]
        image_ = image_[None, :]
        label_ = self.label[idx]
        return image_, label_

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 5)   # num of class

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)  # seed ?
        x = self.fc2(x)
        return F.log_softmax(x)


def train(epoch, network, trainloader, optimizer, y_tr, log_interval, para = 10):
    train_losses = []
    train_counter = []

    i = 0
    network.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data.requires_grad_(True)
        
        optimizer.zero_grad()
        output = network(data)
        
        second_derivative_loss = 0
        for r in range(len(data)):    # iterate all batches
            for j in range(5):
                    if j != y_tr[i]:
                        def f(sample):
                                res = network(sample)
                            #     print(res.shape) [1, 10]
                                res = res.view(5)
                                return res[j]
                        H_matrix = torch.autograd.functional.hessian(f, data[r])
                    #     print(H_matrix.shape)  [1, 28, 28, 1, 28, 28]
                        H_matrix = H_matrix.view(28, 28, 28, 28)
                        trace = 0
                        for a in range(28):
                                for b in range(28):
                                    trace += abs(H_matrix[a][b][a][b])  #abs
                        second_derivative_loss += trace
                    
        
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        

        second_derivative_loss = para*second_derivative_loss
        loss += second_derivative_loss
        loss.backward(retain_graph=True)
        
        

        optimizer.step()
        if batch_idx % log_interval == 0:
            print("derivative loss:", second_derivative_loss)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(trainloader.dataset)))
            
    return network
            
def test(network, valloader):
    test_losses = []
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valloader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(valloader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valloader.dataset),
        100. * correct / len(valloader.dataset)))
    
    
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())
        self.covar_module = gpytorch.kernels.MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def scores(loader, network):
    test_losses = []
    
    network.eval()
    # outputs = [] 
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = network(data)
            # outputs.append(output)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    return output   

def model_initialize_GP(models, likelihoods, mlls, opts, all_index, train_x, log_y, lr, n):
    models, likelihoods, mlls, opts = [], [], [], []

    for j in range(5):
        mo, l, ml, o = [], [], [], []
        for k in range(5):
            val_index = np.arange(np.floor(n/5)*k, np.floor(n/5)*(k+1))
            train_index = [i for i in all_index if i not in val_index]
            x = train_x[train_index, :]
            y = log_y[train_index, j]
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(x, y, likelihood)
            likelihood.train()
            model.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            l.append(likelihood)
            mo.append(model)
            ml.append(mll)
            
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)  # Includes GaussianLikelihood parameters
            o.append(optimizer)
            
        models.append(mo)
        likelihoods.append(l)
        mlls.append(ml)
        opts.append(o)
    
    return models, likelihoods, mlls, opts
        
        
def train_GP(train_x, all_index, log_y, likelihoods, models, mlls, opts, pred_train_y, n):
    training_iter = 10
    # train_y = torch.load('train_score.pt') : x axis

    for j in range(5):  # j represents the 10 labels
        p = []
        for k in range(5): # k represents the k fold cross validation
            print("Model", j, ", Cross Validation Group", k)
            # initialize the model
            val_index = np.arange(np.floor(n/5)*k, np.floor(n/5)*(k+1))
            train_index = [i for i in all_index if i not in val_index]
            tr_x = train_x[train_index, :]
            val_x = train_x[val_index, :]
            tr_y = log_y[train_index, j]
            # val_y = train_y[val_index, k]
            likelihood = likelihoods[j][k]
            model = models[j][k]
            mll = mlls[j][k]
            optimizer = opts[j][k]
            
            # training
            for i in range(training_iter):                
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(tr_x)
                # print(output.shape, tr_y.shape)
                
                # Calc loss and backprop gradients
                loss = -mll(output, tr_y)
                loss.backward()
                
                if i % 4 == 0:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        #model.covar_module.base_kernel.lengthscale.item(),
                        0.1,
                        model.likelihood.noise.item()
                    ))
                optimizer.step()
            
                # evaluation of the current model
            model.eval()
            likelihood.eval()
            with gpytorch.settings.fast_pred_var(), torch.no_grad():
                test_dist = model(val_x)
                pred_means = test_dist.loc
                p.append(pred_means)
        p = torch.stack(p)
        pred_train_y.append(p)
    
    return pred_train_y

def main():
    
    '''Dataset'''
    (x_tr, y_tr), (x_ts, y_ts) = keras.datasets.mnist.load_data()
    # selecting only label 0 to 5
    x, y = [], []
    for i in range(len(x_tr)):
        if y_tr[i] < 5:
            x.append(x_tr[i])
            y.append(y_tr[i])
    x_tr, y_tr = torch.tensor(x), torch.tensor(y)
    print("x_tr", x_tr.shape)
    x, y = [], []
    for i in range(len(x_ts)):
        if y_ts[i] < 5:
            x.append(x_ts[i])
            y.append(y_ts[i])
    x_ts, y_ts = torch.tensor(x), torch.tensor(y)

    x_tr, y_tr = x_tr[0:6400], y_tr[0:6400]
    x_ts, y_ts = x_ts[0:640], y_ts[0:640]
    trainset = MNISTDataset(x_tr, y_tr)
    valset = MNISTDataset(x_ts, y_ts)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)
    # dataiter = iter(trainloader)
    
    '''Model Setup'''
    n_epochs = 6
    learning_rate = 0.006
    momentum = 0.5
    log_interval = 5

    networks, opts = [], []

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)
    
    
    """GP model"""
    with open('features_UMAP.npy', 'rb') as f:
        train_x = np.load(f)
        train_y_label = np.load(f)
        test_x = np.load(f)   
        test_y = np.load(f)
    
    train_x, train_y_label, test_x, test_y = \
        torch.from_numpy(train_x), torch.from_numpy(train_y_label), \
            torch.from_numpy(test_x), torch.from_numpy(test_y)
     
    # selecting only label 0 to 5
    x, y = [], []
    for i in range(len(train_x)):
        if train_y_label[i] < 5:
            x.append(train_x[i])
    train_x = torch.stack(x)
    print("train_x", train_x.shape)
    x, y = [], []
    for i in range(len(test_x)):
        if test_y[i] < 5:
            x.append(test_x[i])
            y.append(test_y[i]) 
    test_x, test_y = torch.stack(x), torch.stack(y)       
            
    '''Model Setup'''
    n = 2000
    all_index = np.arange(0, n)
    lr = 0.1
    
    
    '''Main Training'''
    test(network, valloader)
    parameters = [10]
    # parameters = [0.1, 1, 10, 100]
    for para in parameters:
                
        network = Net()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)
        for epoch in range(1, n_epochs + 1):
            network = train(epoch, network, trainloader, optimizer, y_tr, log_interval, para)
            test(network, valloader)
            
        trloader = torch.utils.data.DataLoader(trainset, batch_size=6400, shuffle=False)
        # tsloader = torch.utils.data.DataLoader(valset, batch_size=640, shuffle=False)
        train_score = scores(trloader, network)
        # test_score = scores(tsloader)
        filename = "data/para" + str(para) + '_train_score.pt'
        torch.save(train_score, filename)
        # torch.save(test_score, 'test_score.pt')
        
        train_y = train_score  # no more modifition needed
        
        # using logy = x
        train_x, train_y = train_x[0: n], train_y[0: n, :]
        test_y = test_y.to(torch.int64)
        add = - torch.min(train_y) + 1
        log_y = add + train_y
        log_y = torch.log(log_y)
        
        
        models, likelihoods, mlls, opts = \
            model_initialize_GP(all_index, train_x, log_y, lr, n)
        pred_train_y = []
        pred_train_y = train_GP(train_x, all_index, log_y, likelihoods, models, mlls, opts, pred_train_y, n)
        
        
        for j in range(5):
            pred_train_y[j] = pred_train_y[j].view(1, -1).flatten()
            pred_train_y[j] = torch.exp(pred_train_y[j]) - add
            
        for j in range(5):
            name = "plots/para" + str(para) + "scaled_model" + str(j) +".png"
            x_data = train_y[:, j]
            mini = torch.min(x_data)
            if mini < -20:
                mini = -40
            else: 
                mini = -20
            y_data= pred_train_y[j]
            plt.plot(x_data, y_data, 'o', color='black')
            x = np.linspace(-10,2,100)
            y = x
            plt.plot(x, y, '-r', label='y=2x+1')
            plt.title('Model' + str(j))
            plt.xlabel('DNN training scores', color='#1C2833')
            plt.ylabel('GP training scores', color='#1C2833')
            plt.savefig(name, facecolor = 'white', transparent = False)
            plt.show()
    
if __name__ == "__main__":
    main()
