# -*- coding: utf-8 -*-
'''
This module only defines the Hyperpars class that acts like a dictionary, but
that I find practical in this case.
'''

import torch
from torch import nn

class Hyperpars:
    '''
    This class contains parameters to be used by several other classes in this project.
    '''

    def __init__(self, criterion=[nn.BCELoss, True], optimizer=torch.optim.Adam, batch_size=100,
                 learning_rate=0.001, threshold=0.6, image_size=[256,256], numberoflabels=8,
                 train_size=19323, val_size=1000, test_size=1000, epochs=1, activation=nn.Sigmoid(),
                 logfile='./funduslog.txt', numberofchannels=3):
        self.learning_rate=learning_rate
        self.criterion=criterion[0]
        self.needsactivation=criterion[1] # True, if there is no activation fct embedded in the loss fct.
        self.batch_size=batch_size
        self.threshold=threshold # Threshold value to round up to a one, i.e. a positive diagnosis
        self.optimizer=optimizer
        self.image_size=image_size
        if image_size[0]%8 != 0 or image_size[1]%8 != 0:
            raise Exception('Image size should be divisible by 8')
        self.numberoflabels=numberoflabels
        # Number of images in each of the datasets
        self.train_size=train_size
        self.val_size=val_size
        self.test_size=test_size
        self.epochs=epochs
        self.activation=activation
        self.logfile=logfile
        self.numberofchannels=numberofchannels
