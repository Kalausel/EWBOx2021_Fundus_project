# -*- coding: utf-8 -*-
"""
This script uses the provided classes to train and test the model.
"""

from matplotlib import pyplot as plt
from datetime import datetime
from torchvision import transforms
from Hyperpars import Hyperpars
from DataPrep import DataPrep
from Doctor import Doctor
from Model1 import Model

# Leave out the following images
invalid_images=['2174_right.jpg','2175_left.jpg','2176_left.jpg','2177_left.jpg',
                '2177_right.jpg','2178_right.jpg','2179_left.jpg','2179_right.jpg',
                '2180_left.jpg','2180_right.jpg','2181_left.jpg','2181_right.jpg',
                '2182_left.jpg','2182_right.jpg','2957_left.jpg','2957_right.jpg',
                '1706_left.jpg','1710_right.jpg','4580_left.jpg']

img_dir='./Small_multiplied_images'
labels_path='./training_annotations.csv'
hyperpars=Hyperpars(epochs=4, learning_rate=0.000000005)
hyperpars.test_size=hyperpars.test_size-len(invalid_images) # Adjust test_size to the data without the invalid images.
# Thresholds for parameters from training with BCELoss. Found by trial and error
hyperpars.threshold=[0.34448990678432173, 0.343063769811366, 0.5150682767242185,
                     0.8933555631822088, 0.6065823723591031, 0.34869314948235525,
                     1.0137435401083779, 0.2807865752721815]

# Transform square and resize
def square_and_resize(image):
    '''
    This transform is applied to all the images to make them reasonably equal.
    Not needed in the current usage, since it has already been applied to the 
    saved images.
    '''
    length1=image.size()[1]
    length2=image.size()[2]
    length=min(length1,length2)
    if abs(length1-length2)>50:
        transf=transforms.Compose([transforms.CenterCrop(int(length*1.1)),
                                   transforms.Resize(hyperpars.image_size)])
    else:
        transf=transforms.Compose([transforms.CenterCrop(int(length)),
                                   transforms.Resize(hyperpars.image_size)])
    return transf(image)

random_jitter=transforms.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.2, hue=0.01)

dataprep=DataPrep(hyperpars, img_dir, labels_path, transform=random_jitter ,
                  flip_right=False, skip_images=invalid_images,
                   label_multiplier=2)
model=Model(hyperpars)
doctor=Doctor(model, hyperpars)
train_dataloader, val_dataloader, test_dataloader = dataprep.data_loaders()

# weights=None to start from scratch; Run whole script!
# weights='current' to continue training the same Doctor; Run only part of script (do not reinitialize Doctor!)
# If ran whole script by accident, can input the weights specifically to keep training, if they have been saved.
tracking, loss_list = doctor.training(train_dataloader, val_dataloader, weights='02032022_212742')
correct_ratio=doctor.test(test_dataloader, weights='last_saved')

batch=tracking.copy()
for j in range(len(tracking)):
    batch[j]=int(tracking[j][0]*hyperpars.train_size/hyperpars.batch_size+tracking[j][1])
plt.figure()
plt.plot(batch, loss_list, label='Loss vs batch')
plt.xlabel='batch'
plt.ylabel='loss'
now=datetime.now().strftime("%d%m%Y_%H%M%S")
plt.savefig('./Loss_plots/'+now+'.png')

'''
# Display torch tensor using
plt.imshow(image.permute([1,2,0]))
'''
