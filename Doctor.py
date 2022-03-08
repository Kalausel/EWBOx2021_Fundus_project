# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:48:07 2022

@author: Klaus
"""

import os
import statistics
from datetime import datetime
from shutil import copyfile
import torch

class Doctor:
    '''
    The Doctor class contains methods to perform training  of a predefined ML model
    and uses this model to diagnose images.
    Some methods are only used for training in google colabs, but they are not executed,
    if you don't explicitly set colab=True in the training or testing method.
    '''
    def __init__(self, model, hyperpars):
        self.model=model
        self.hyperpars=hyperpars
        self.weights=None
        self.completed_epochs=0

    def training(self, train_loader, val_loader, weights='last_saved', colab=False):
        '''
        This method changes the parameters of self.model using self.hyperpars.optimizer.
        It saves the best weights in the folder './Weights', logs its activities and
        outputs the loss for each batch.
        Input: Dataloader types containing training and validation data
        Output: tracking: list of [epoch number, batch number] for each batch
                loss_list: list of loss for each batch.
        '''

        self.log('\n\n\n'+datetime.now().strftime("%d/%m/%Y_%H:%M:%S")+' START TRAINING; weights='
                 +str(weights)+'; learning_rate='+str(self.hyperpars.learning_rate)) # Log

        # Initialisations
        best_loss_val = float("inf") # Make sure that first weights are saved in the validation step.
        optimizer=self.hyperpars.optimizer(self.model.parameters(),lr=self.hyperpars.learning_rate)
        tracking=[] # keeps track of epochs and batch number
        loss_list=[] # keeps track of the loss for each batch
        checkpoint=None

        # The training method uses self.weights throughout. It is set here.
        if weights=='current':
            # Those are the weights already saved in self.model (random, if it has not yet been trained)
            previous_epochs=self.completed_epochs
        elif weights is None: # start from scratch
            self.model.weight_reset()
            previous_epochs=0
        else:
            if weights=='last_saved': # Doctor still has self.weights from last call of training()
                if self.weights is None:
                    raise Exception('Last saved weights not available in Doctor class.'+
                                    ' E.g. if these weights have been saved by another'+
                                    ' instance of the Doctor class')
            elif weights: # If specific weights are given, use those
                self.weights=weights
            checkpoint = torch.load("./Weights/weights_"+self.weights+".pt")
            self.model.load_state_dict(checkpoint['model_state_dict']) # Load those specified weights
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint: # If training is continued on previously trained weights
            previous_epochs=checkpoint['previous_epochs'] # continue at the previous epoch number

        self.model.train() # set model to training mode
        # Loop through epochs
        for epoch in range(previous_epochs, previous_epochs+self.hyperpars.epochs):

            self.log('\n    '+datetime.now().strftime("%d/%m/%Y_%H:%M:%S")+' Start epoch '+str(epoch)+
                     ' of '+str(previous_epochs+self.hyperpars.epochs)) # Log

            # TRAINING IN THE NARROW SENSE
            # Loop through the single batches
            for k, (images,labels) in enumerate(train_loader):
                print('Batch number k = '+str(k)+' of '+str(int(self.hyperpars.train_size/self.hyperpars.batch_size)))
                # Forward pass through model only on images
                outputs=self.model(images) # automatically calls forward()-method of the model
                # outputs has torch.size (batch_size x 8), where 8 = number of different disease categories
                if self.hyperpars.needsactivation is True: # Some loss criterions have inbuilt activation, some don't
                    outputs=self.hyperpars.activation(outputs)
                loss=self.hyperpars.criterion()(outputs, labels) # returns Pytorch tensor (of size 1)
                print('Loss = '+str(loss.item()))

                # Storing training data for output
                tracking.append([epoch, k])
                loss_list.append(loss.item())

                # Updating the model
                optimizer.zero_grad() # Reset optimizer
                loss.backward() # The actual back-propagation; Computes gradients
                optimizer.step() # Optimizer training step; Uses gradients to update self.model.parameters()

            # VALIDATION
            self.log('\n    '+datetime.now().strftime("%d/%m/%Y_%H:%M:%S")+' Start validation') # Log
            self.model.eval() # set model to evaluation mode
            running_loss_val=0.0 # Reset running_loss from last epoch's validation
            for k, (images,labels) in enumerate(val_loader):
                print('Batch number k = '+str(k)+' of '+str(int(self.hyperpars.val_size/self.hyperpars.batch_size)))
                outputs=self.model(images) # forward pass through model
                if self.hyperpars.needsactivation is True: # Some loss criterions have inbuilt activation, some don't
                    outputs=self.hyperpars.activation(outputs)
                loss=self.hyperpars.criterion()(outputs,labels) # Compute the loss of this batch
                print('Loss = '+str(loss.item()))
                # Add the loss to the running loss of this epoch's validation
                running_loss_val+=loss.item()/len(val_loader)

            self.log(text='\n    '+datetime.now().strftime("%d/%m/%Y_%H:%M:%S")+' Finished epoch '
                     +str(epoch)+'; Validation loss='+str(running_loss_val)) # Log
            self.completed_epochs=epoch+1

            # If the current model behaves best on the val data, save the model parameters from the last epoch
            if running_loss_val<best_loss_val:
                if epoch>previous_epochs: # Do not remove weights from previous runs! Running loss is not compared to previous runs!
                    os.remove('./Weights/weights_'+self.weights+'.pt') # Delete the weight files, which are not the best
                    print('\n    '+datetime.now().strftime("%d/%m/%Y_%H:%M:%S")+' Deleted weights_'+self.weights)
                    if colab:
                        # The below function moves the file to drive's trash bin as file of size 0 Bytes
                        self.delete_in_drive('/content/drive/MyDrive/Colab Notebooks/EWBOxProject/Weights/weights_'+self.weights+'.pt')
                # Save the model's parameters as a .pt-file
                now=datetime.now().strftime("%d%m%Y_%H%M%S")
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'previous_epochs': epoch+1},
                            './Weights/weights_'+now+'.pt')
                self.weights=now
                # Update best_loss_val for comparison with the next batch's running_loss
                best_loss_val=running_loss_val

                self.log('\n    '+str(now)+' Saved new weights'+'; Validation loss='+str(best_loss_val))
            if colab:
                self.copy_to_drive()
        self.log('\n    '+datetime.now().strftime("%d/%m/%Y_%H:%M:%S")+' Finished training') # Log
        if colab:
            self.copy_to_drive()

        return tracking, loss_list # tracking = list of epochs and batches; loss_list = loss for each batch


    def diagnose(self, image, weights='current', transform=None):
        '''
        This method diagnoses a single image.
        Input: Image of a fundus; type torch tensor of size (8)
        Output: Diagnosis; type torch.tensor of size (1,8)
        '''
        if transform: # if self.transform != None
            image=transform(image) # apply this transform
        if weights=='current':
            pass # use the weights that are already in self.model
        else: # else, load the weights from the input argument
            if weights=='last_saved':
                if self.weights is None:
                    raise Exception('Last saved weights not available in Doctor class. E.g. if these weights have been saved by another instance of the Doctor class')
                weights=self.weights # self.weights are the previously saved weights
            checkpoint = torch.load("./Weights/weights_"+weights+".pt")
            self.model.load_state_dict(checkpoint['model_state_dict']) # Load weights
        self.model.eval() # set model to evaluation mode
        output=self.model(image[None,:,:,:])
        if self.hyperpars.needsactivation is True: # Some models have inbuilt activation, some don't
            output=self.hyperpars.activation(output)
        prediction=self.threshold(output) # Now, round the output to give either 1 or 0 in every entry
        return prediction

    def test(self, test_loader, weights='current', colab=False): # Default weights are self.weights
        '''
        Input: Dataloader type containing test data
        Output: float type: ratio of correct diagnoses to total diagnoses.
        '''

        # The test method uses weights, not self.weights. weights is assigned below.
        if weights=='current':
            pass # the current self.model still has the current weights inside
        else:
            if weights=='last_saved': # self.weights are the previously saved weights.
                if self.weights is None:
                    raise Exception('Last saved weights not available in Doctor class. E.g. if these weights have been saved by another instance of the Doctor class')
                weights=self.weights
            # if specific weights are given as an argument, load those
            checkpoint = torch.load("./Weights/weights_"+weights+".pt")
            self.model.load_state_dict(checkpoint['model_state_dict']) # Load weights

        self.log('\n\n'+datetime.now().strftime("%d/%m/%Y_%H:%M:%S")+' START TESTING; weights='+str(weights)) # Log

        self.model.eval() # set model to evaluation mode

        # no_grad makes the evaluation quicker. Only to be used in testing, hence use "with" to make sure that it's not used outside
        with torch.no_grad():
            # Predict the test images and compare to their labels.
            correct_ratio_batches=[]
            for k, (images,labels) in enumerate(test_loader): # Iterate through test_data one batch at a time
                print('Batch number k = '+str(k)+' of '+str(int(self.hyperpars.test_size/self.hyperpars.batch_size)))
                model_output=self.model(images)
                if self.hyperpars.needsactivation is True: # Some models have inbuilt activation, some don't
                    model_output=self.hyperpars.activation(model_output)
                predictions=self.threshold(model_output) # Now, round the output to give either 1 or 0 in every entry
                # Renormalize labels, if they have been multiplied by a factor
                if labels.max()!=0:
                    labels=labels/labels.max()
                labels=labels.int()
                correct_predictions=(predictions==labels) # for individual diseases
                # The next line gives True for every row where all diseases are correctly predicted.
                correct_predictions=correct_predictions.min(axis=1).values # For each row (=image), if one entry is false, return false for entire row
                # correct_predictions is now a torch.Tensor of shape (1,batch_size) with True or False for each element, i.e. for each image in the batch.
                # Now, append the ratio of correct predictions (prediction is correct for one eye, if all 8 bools are correct) from this batch to the below list.
                correct_ratio_batch=torch.sum(correct_predictions).item()/correct_predictions.shape[0]
                print('Correct_ratio of batch = '+str(correct_ratio_batch))
                correct_ratio_batches.append(correct_ratio_batch)
        correct_ratio=statistics.mean(correct_ratio_batches) # the total ratio of correct predictions (avgd over all batches)

        self.log('\n    '+datetime.now().strftime("%d/%m/%Y_%H:%M:%S")+
                 ' Finished testing; correct_ratio='+str(correct_ratio)) # Log

        if colab:
            self.copy_to_drive()

        return correct_ratio

    def log(self, text):
        '''
        This function writes the input string text into the logfile.
        '''
        with open(self.hyperpars.logfile, 'a') as file:
            file.write(text)
        print(text)

    def threshold(self, x, threshold=None): # x is a whole batch!
        '''
        This function sets the model outputs to 0 and 1 depending on the given threshold.
        If no category reaches the threshold, it sets the max to 1.
        If 'normal fundus' and another category reach threshold, it chooses
        'normal fundus' iff this is the maximum.
        Input:  x:          torch.tensor of size (batch_size, 8)
                treshold:   float, list of length 8 or tensor of size (1,8)

        Output:             torch.tensor of size (batch_size, 8) containing 0s and 1s
        '''
        if threshold is None:
            threshold=torch.Tensor(self.hyperpars.threshold).repeat(x.shape[0],1)
        output=(x>threshold)
        zero_indicator=torch.sum(output,1) # if no category has reached the threshold, this is 0 at that number in the batch
        for index, value in enumerate(zero_indicator): # index=number within batch
            if value==0:
                output[index]=(x[index]>=torch.max(x[index])) # set maximum to 1, others to 0
            if output[index][0] == 1 and torch.sum(output[index]) >= 1.5: # if 'normal' and something else is positive; torch does some minor rounding sometimes, hence 1.5
                if torch.max(x[index]) == x[index][0]: # if 'normal fundus ' is the maximum
                    output[index]=torch.Tensor([1,0,0,0,0,0,0,0]) # choose 'normal fundus'
                else: # otherwise
                    output[index][0]=0 # choose everything except 'normal fundus'
        return output.int()

    def copy_to_drive(self):
        '''
        This function is used to copy altered files from colab to drive.
        It is only used when training in colab.
        '''
        drive_folder='/content/drive/MyDrive/Colab Notebooks/EWBOxProject/'
        colab_folder='/content/EWBOxProject/'

        # Remove old logfile and copy the new one.
        if os.path.isfile(os.path.join(drive_folder,'funduslog.txt')):
            os.remove(os.path.join(drive_folder,'funduslog.txt'))
        copyfile(os.path.join(colab_folder,'funduslog.txt'),os.path.join(drive_folder,'funduslog.txt'))
        print('Copied funduslog')

        # Copy all the weights files that are not yet in drive.
        weights_in_colab=os.listdir(os.path.join(colab_folder,'Weights'))
        weights_in_drive=os.listdir(os.path.join(drive_folder,'Weights'))
        for file in weights_in_colab:
            if file in weights_in_drive:
                pass
            else:
                copyfile(os.path.join(colab_folder,'Weights',file), os.path.join(drive_folder,'Weights',file))
                print('Copied weights '+str(file))

        # Copy all the loss plots that are not yet in drive.
        plots_in_colab=os.listdir(os.path.join(colab_folder,'Loss_plots'))
        plots_in_drive=os.listdir(os.path.join(drive_folder,'Loss_plots'))
        for file in plots_in_colab:
            if file in plots_in_drive:
                pass
            else:
                copyfile(os.path.join(colab_folder,'Loss_plots',file), os.path.join(drive_folder,'Loss_plots',file))
                print('Copied loss plots '+str(file))

    def delete_in_drive(self, file):
        '''
        This function moves file to drive's trash bin as file of size 0 Bytes.
        This prevents drive running out of storage space due to files in the trash bin.
        '''
        open(file, 'w').close() # overwrite with empty file
        os.remove(file) # move to drive trash bin
        print('\n    '+datetime.now().strftime("%d/%m/%Y_%H:%M:%S")+'Deleted from drive: '+file)
    