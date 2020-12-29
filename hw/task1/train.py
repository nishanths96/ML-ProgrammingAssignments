import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from data import get_data_loader
from network import Network
from config import cfg

try:
    from termcolor import cprint
except ImportError:
    cprint = None

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def get_lr(optimizer):
    # Returns the current Learning Rate being used by the optimizer
    return optimizer.param_groups[0]['lr']


'''
Use the average meter to keep track of average of the loss or 
the test accuracy! Just call the update function, providing the
quantities being added, and the counts being added
'''


class AvgMeter():
    def __init__(self):
        self.qty = 0
        self.cnt = 0

    def update(self, increment, count):
        self.qty += increment
        self.cnt += count

    def get_avg(self):
        if self.cnt == 0:
            return 0
        else:
            return self.qty / self.cnt


def run(net, epoch, loader, optimizer, criterion, logger, scheduler, train=True):
    global device, train_x_i, test_x_i

    # if train:
    #     x_axis = np.linspace(epoch, epoch+1, int(np.ceil(60000/cfg['batch_size'])))
    #     x_i = 0
    # else:
    #     x_axis = np.linspace(epoch, epoch + 1, int(np.ceil(10000 / cfg['batch_size'])))
    #     x_i = 0

    # Initalize the different Avg Meters for tracking loss and accuracy (if test)
    running_loss = AvgMeter()
    accuracy = AvgMeter()

    # Performs a pass over data in the provided loader
    for images, labels in loader:
        # flatten the input image
        images = images.view(images.shape[0], -1)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if train:
            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(images.float())
            loss = criterion(outputs, labels.long())
            # backward propagation
            loss.backward()
            optimizer.step()

            # Iterate over the loader and find the loss. Calculate the loss and based on which
            # set is being provided update you model. Also keep track of the accuracy if we are running
            # on the test set.
            running_loss.update(loss.item(), images.shape[0])
            logger.add_scalar("Loss/train", loss.item(), train_x_i)
            train_x_i += 1
        else:
            outputs = net(images.float())
            loss = criterion(outputs, labels.long())
            running_loss.update(loss.item(), images.shape[0])

            _, predicted = torch.max(outputs.data, 1)
            accuracy.update((predicted == labels).sum().item(), labels.size(0))
            # Log the training/testing loss using tensorboard.
            logger.add_scalar("Loss/test", loss.item(), test_x_i)
            logger.add_scalar("Accuracy/test", accuracy.get_avg(), test_x_i)
            test_x_i += 1

    # return the average loss, and the accuracy (if test set)
    if train:
        return running_loss.get_avg(), 0
    else:
        return running_loss.get_avg(), 100 * accuracy.get_avg()


def train(net, train_loader, test_loader, logger):
    # Define the SGD optimizer here. Use hyper-parameters from cfg
    optimizer = optim.SGD(net.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], nesterov=cfg['nesterov'],
                          weight_decay=cfg['weight_decay'])

    # Define the criterion (Objective Function) that you will be using
    criterion = nn.CrossEntropyLoss()

    # Define the ReduceLROnPlateau scheduler for annealing the learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg['patience'], factor=cfg['lr_decay'])
    weight_visualization1 = []
    weight_visualization2 = []
    for i in range(cfg['epochs']):
        # Run the network on the entire train dataset. Return the average train loss
        # Note that we don't have to calculate the accuracy on the train set.
        loss, _ = run(net, i, train_loader, optimizer, criterion, logger, scheduler)

        # Get the current learning rate by calling get_lr() and log it to tensorboard
        logger.add_scalar("LearningRate/train", get_lr(optimizer), i)

        # Logs the training loss on the screen, while training
        if i % cfg['log_every'] == 0:
            log_text = "Epoch: [%d/%d], Training Loss:%2f" % (i, cfg['epochs'], loss)
            log_print(log_text, color='green', attrs=['bold'])

        # Evaluate our model and add visualizations on tensorboard
        if i % cfg['val_every'] == 0:
            # Run the network on the test set, and get the loss and accuracy on the test set
            loss, acc = run(net, i, test_loader, optimizer, criterion, logger, scheduler, train=False)
            log_text = "Epoch: %d, Test Accuracy:%2f" % (i, acc)
            log_print(log_text, color='red', attrs=['bold'])

            # Perform a step on the scheduler, while using the Accuracy on the test set
            scheduler.step(acc)

            # Use tensorboard to log the Test Accuracy and also to perform visualization of the
            # 2 weights of the first layer of the network!
            weight_visualization1.append(net.layer1.weight.data[51].reshape((1, 28, 28)))
            weight_visualization2.append(net.layer1.weight.data[92].reshape((1, 28, 28)))

    grid1 = make_grid(weight_visualization1, nrow=4, normalize=True)
    grid2 = make_grid(weight_visualization2, nrow=4, normalize=True)
    logger.add_image('Visualization1/Node_50', grid1, 0)
    logger.add_image('Visualization2/Node_80', grid2, 0)
    logger.add_image('Visualization2/Node_80', grid2, 0)


def get_device():
    if torch.cuda.is_available():
        # device = 'cpu'
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


device = get_device()
train_x_i = 0
test_x_i = 0

if __name__ == '__main__':
    # Create a network object
    net = Network().to(device)

    # Create a tensorboard object for logging
    writer = SummaryWriter('runs/fashion_mnist_1')

    # Create train data loader
    train_loader = get_data_loader("train", augmentation=True)

    # Create test data loader
    test_loader = get_data_loader("test")

    # Run the training!
    train(net, train_loader, test_loader, writer)