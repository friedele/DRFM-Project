#### Data Analysis Utilities ####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import sys
import time
import torch
import torch.nn as nn
import torch.nn.init as init

import shutil

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
_, term_width = shutil.get_terminal_size()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def horizontal_plots_loss(loss1, loss2, ds_path, algo1="", algo2=""):
    max_y = max([max(loss1), max(loss2)])
    max_y += max_y * 0.05

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    alg1 = sns.lineplot(ax=ax[0], data=loss1)
    alg1.set_ylim([0, max_y])
    alg1.set(title=algo1, xlabel='Iteration', ylabel='Loss')

    alg2 = sns.lineplot(ax=ax[1], data=loss2)
    alg2.set_ylim([0, max_y])
    alg2.set(title=algo2, xlabel='Iteration', ylabel='Loss')

    plt.show()

    fig.savefig("{}/horizontal_losses.png".format(ds_path))


def plot_single_loss(loss, ds_path, algo):
    sns.set_theme(style="darkgrid")

    ax = sns.lineplot(data=loss, dashes=False, legend=False)
    ax.set(title=algo, xlabel='Iteration', ylabel='Loss')

    plt.show()
    ax.get_figure().savefig("{}/loss_{}.png".format( ds_path, algo))


def plot_losses(loss1, loss2, ds_path, algo1="", algo2=""):
    sns.set_theme(style="darkgrid")
    np.seterr(all='raise')

    ax = sns.lineplot(data=loss1, dashes=False)
    ax = sns.lineplot(data=loss2, dashes=False)
    ax.set(xlabel='Iteration', ylabel="Loss")

    plt.legend(loc='upper right', labels=['LBFGS', 'Momentum Descent'])
    plt.show()
    ax.get_figure().savefig("{}/losses_in_single_plot.png".format(ds_path))


def plot_convergence_rates(loss1, loss2, ds_path, algo1="", algo2=""):
    sns.set_theme(style="darkgrid")
    np.seterr(all='raise')
    rates1 = []
    d = np.abs(loss1 - loss1[-1])
    for i in range(len(d) - 1):
        try:
            rates1.append(np.log(d[i + 1]) / np.log(d[i]))
        except:
            rates1.append(1)
    rates2 = []
    d = np.abs(loss2 - loss2[-1])
    for i in range(len(d) - 1):
        try:
            rates2.append(np.log(d[i + 1]) / np.log(d[i]))
        except:
            rates2.append(1)
    ax = sns.lineplot(data=rates1, dashes=False)
    ax = sns.lineplot(data=rates2, dashes=False)
    ax.set(title="Convergence rate", xlabel='Iteration', ylabel="Convergence rate")

    plt.legend(loc='upper right', labels=['LBFGS', 'Momentum Descent'])
    plt.show()
    ax.get_figure().savefig("{}/convergence_rate.png".format(ds_path))

def plot_single_convergence_rate(loss, ds_path, algo="" ):
    sns.set_theme(style="darkgrid")
    np.seterr(all='raise')
    rates = []
    d = np.abs(loss - loss[-1])
    for i in range(len(d) - 1):
        try:
            rates.append(np.log(d[i + 1]) / np.log(d[i]))
        except:
            rates.append(1)

    ax = sns.lineplot(data=rates, dashes=False)
    ax.set(title="Convergence rate of {}".format(algo), xlabel='Iteration', ylabel="Convergence rate")
    plt.show()
    ax.get_figure().savefig("{}/convergence_rate_{}.png".format(ds_path, algo))


def print_results_to_table(model1, model2, ds_path):
    f_stars = [
        model1.loss[-1],
        model2.loss[-1]
    ]
    norms_of_gradient = [
        model1.norm_of_gradient[-1],
        model2.norm_of_gradient[-1]
    ]
    n_of_iterations = [
        model1.optimization_algorithm.iteration,
        model2.optimization_algorithm.iteration
    ]

    total_times = [
        model1.optimization_algorithm.total_time,
        model2.optimization_algorithm.total_time
    ]

    iteration_time_mean = [
        np.mean(model1.time),
        np.mean(model2.time)
    ]

    optimizers = [
        "L-BFGS",
        "MGD"
    ]

    stop_reasons = [
        model1.optimization_algorithm.stop_because_of,
        model2.optimization_algorithm.stop_because_of,
    ]
    df = pd.DataFrame({'opt':optimizers,
                       'f *':f_stars,
                       '||g_k||': norms_of_gradient,
                       '# iterations': n_of_iterations,
                       'total time (s)': total_times,
                       'mean time per it. (s)':iteration_time_mean,
                       'stop reason': stop_reasons})
    df = df.set_index('opt')

    print(df.to_markdown())

    df.to_csv('{}/results_comparison.csv'.format(ds_path))

    # Write also directly latex table to file
    with open('{}/results_comparison_latex.txt'.format(ds_path), 'w') as f:
        f.write(str(df.to_latex()))


def print_results_to_table_array_of_model(models, ds_path, optimizers):
    f_stars = []
    norms_of_gradient = []
    n_of_iterations = []
    total_times = []
    iteration_time_mean = []
    stop_reasons = []

    for m in models:
        f_stars.append(m.loss[-1])
        norms_of_gradient.append(m.norm_of_gradient[-1])
        n_of_iterations.append(m.optimization_algorithm.iteration)
        total_times.append(m.optimization_algorithm.total_time)
        iteration_time_mean.append(np.mean(m.time))
        stop_reasons.append(m.optimization_algorithm.stop_because_of)

    df = pd.DataFrame({'opt': optimizers,
                       'f *': f_stars,
                       '||g_k||': norms_of_gradient,
                       '# iterations': n_of_iterations,
                       'total time (s)': total_times,
                       'mean time per it. (s)': iteration_time_mean,
                       'stop reason': stop_reasons})
    df = df.set_index('opt')

    df.to_csv('{}/results_comparison.csv'.format(ds_path))
    # Write also directly latex table to file
    with open('{}/results_comparison_latex.txt'.format(ds_path), 'w') as f:
        f.write(str(df.to_latex()))

    df.sort_values(['f *', '# iterations'], ascending=[True, True], inplace=True )

    df.to_csv('{}/results_comparison_sorted.csv'.format(ds_path))
    # Write also directly latex table to file
    with open('{}/results_comparison_sorted_latex.txt'.format(ds_path), 'w') as f:
        f.write(str(df.to_latex()))

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

#%% Compute Statistics (Training Loss, Test Loss, Test Accuracy)

def compute_stats(X_train, y_train, X_test, y_test, opfun, accfun, ghost_batch=128):
    """
    Computes training loss, test loss, and test accuracy efficiently.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 8/29/18.

    Inputs:
        X_train (nparray): set of training examples
        y_train (nparray): set of training labels
        X_test (nparray): set of test examples
        y_test (nparray): set of test labels
        opfun (callable): computes forward pass over network over sample Sk
        accfun (callable): computes accuracy against labels
        ghost_batch (int): maximum size of effective batch (default: 128)

    Output:
        train_loss (float): training loss
        test_loss (float): test loss
        test_acc (float): test accuracy

    """

    # compute test loss and test accuracy
    test_loss = 0
    test_acc = 0

    # loop through test data
    for smpl in np.array_split(np.random.permutation(range(X_test.shape[0])), int(X_test.shape[0]/ghost_batch)):

        # define test set targets
        if(torch.cuda.is_available()):
            test_tgts = torch.from_numpy(y_test[smpl]).cuda().long().squeeze()
        else:
            test_tgts = torch.from_numpy(y_test[smpl]).long().squeeze()

        # define test set ops
        testops = opfun(X_test[smpl])

        # accumulate weighted test loss and test accuracy
        if(torch.cuda.is_available()):
            test_loss += F.cross_entropy(testops, test_tgts).cpu().item()*(len(smpl)/X_test.shape[0])
        else:
            test_loss += F.cross_entropy(testops, test_tgts).item()*(len(smpl)/X_test.shape[0])

        test_acc += accfun(testops, y_test[smpl])*(len(smpl)/X_test.shape[0])

    # compute training loss
    train_loss = 0

    # loop through training data
    for smpl in np.array_split(np.random.permutation(range(X_train.shape[0])), int(X_test.shape[0]/ghost_batch)):

        # define training set targets
        if(torch.cuda.is_available()):
            train_tgts = torch.from_numpy(y_train[smpl]).cuda().long().squeeze()
        else:
            train_tgts = torch.from_numpy(y_train[smpl]).long().squeeze()

        # define training set ops
        trainops = opfun(X_train[smpl])

        # accumulate weighted training loss
        if(torch.cuda.is_available()):
            train_loss += F.cross_entropy(trainops, train_tgts).cpu().item()*(len(smpl)/X_train.shape[0])
        else:
            train_loss += F.cross_entropy(trainops, train_tgts).item()*(len(smpl)/X_train.shape[0])

    return train_loss, test_loss, test_acc


#%% Compute Objective and Gradient Helper Function

def get_grad(optimizer, X_Sk, y_Sk, opfun, ghost_batch=128):
    """
    Computes objective and gradient of neural network over data sample.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 8/29/18.

    Inputs:
        optimizer (Optimizer): the PBQN optimizer
        X_Sk (nparray): set of training examples over sample Sk
        y_Sk (nparray): set of training labels over sample Sk
        opfun (callable): computes forward pass over network over sample Sk
        ghost_batch (int): maximum size of effective batch (default: 128)

    Outputs:
        grad (tensor): stochastic gradient over sample Sk
        obj (tensor): stochastic function value over sample Sk

    """

    if(torch.cuda.is_available()):
        obj = torch.tensor(0, dtype=torch.float).cuda()
    else:
        obj = torch.tensor(0, dtype=torch.float)

    Sk_size = X_Sk.shape[0]

    optimizer.zero_grad()

    # loop through relevant data
    for idx in np.array_split(np.arange(Sk_size), max(int(Sk_size/ghost_batch), 1)):

        # define ops
        ops = opfun(X_Sk[idx])

        # define targets
        if(torch.cuda.is_available()):
            tgts = Variable(torch.from_numpy(y_Sk[idx]).cuda().long().squeeze())
        else:
            tgts = Variable(torch.from_numpy(y_Sk[idx]).long().squeeze())

        # define loss and perform forward-backward pass
        loss_fn = F.cross_entropy(ops, tgts)*(len(idx)/Sk_size)
        loss_fn.backward()

        # accumulate loss
        obj += loss_fn

    # gather flat gradient
    grad = optimizer._gather_flat_grad()

    return grad, obj

#%% Adjusts Learning Rate Helper Function

def adjust_learning_rate(optimizer, learning_rate):
    """
    Sets the learning rate of optimizer.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 8/29/18.

    Inputs:
        optimizer (Optimizer): any optimizer
        learning_rate (float): desired steplength

    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return

#%% CUTEst PyTorch Interface
    
class CUTEstFunction(torch.autograd.Function):
    """
    Converts CUTEst problem using PyCUTEst to PyTorch function.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 9/21/18.

    """

    @staticmethod
    def forward(ctx, input, problem):
        x = input.clone().detach().numpy()
        obj, grad = problem.obj(x, gradient=True)
        ctx.save_for_backward(torch.tensor(grad, dtype=torch.float))
        return torch.tensor(obj, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad, None

class CUTEstProblem(torch.nn.Module):
    """
    Converts CUTEst problem to torch neural network module.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 9/21/18.

    Inputs:
        problem (callable): CUTEst problem interfaced through PyCUTEst

    """

    def __init__(self, problem):
        super(CUTEstProblem, self).__init__()
        # get initialization
        x = torch.tensor(problem.x0, dtype=torch.float)
        x.requires_grad_()

        # store variables and problem
        self.variables = torch.nn.Parameter(x)
        self.problem = problem

    def forward(self):
        model = CUTEstFunction.apply
        return model(self.variables, self.problem)

    def grad(self):
        return self.variables.grad

    def x(self):
        return self.variables

