# -*- coding: utf-8 -*-
"""CNN_GAN.ipynb

# Initialization, utilities (no TODOs)
"""

import torch
import torchvision
import torch.nn as nn
import argparse
import PIL
import random

def to_list(img):
    return list(map(int, img.view((28*28,)).tolist()))

SCALE_OFF = 0
SCALE_RANGE = 1
SCALE_01 = 2


def show_image(tens, imgname=None, scale=SCALE_01):
    """
    Show an image contained in a tensor. The tensor will be reshaped properly, as long as it has the required 28*28 = 784 entries.

    If imgname is provided, the image will be saved to a file, otherwise it will be stored in a temporary file and displayed on screen.

    The parameter scale can be used to perform one of three scaling operations:
        SCALE_OFF: No scaling is performed, the data is expected to use values between 0 and 255
        SCALE_RANGE: The data will be rescaled from whichever scale it has to be between 0 and 255. This is useful for data in an unknown/arbitrary range. The lowest value present in the data will be
        converted to 0, the highest to 255, and all intermediate values will be assigned using linear interpolation
        SCALE_01: The data will be rescaled from a range between 0 and 1 to the range between 0 and 255. This can be useful if you normalize your data into that range.
    """
    r = tens.max() - tens.min()
    img = PIL.Image.new("L", (28,28))
    scaled = tens
    if scale == SCALE_RANGE:
        scaled = (tens - tens.min())*255/r
    elif scale == SCALE_01:
        scaled = tens*255
    img.putdata(to_list(scaled))
    if imgname is None:
        img.show()
    else:
        img.save(imgname)

"""# Classification (5 TODOs)"""

# Used for both tasks
loss_fn = torch.nn.BCELoss()

# TODO 1: Choose a digit
digit = 6

# TODO 2: Change number of training iterations for classifier
n0 = 30

# TODO 3
# Change Network architecture of the discriminator/classifier network. It should have 784 inputs and 1 output (0 = fake, 1 = real)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        #iput layer
        self.linear1 = nn.Linear(784,256)
        self.linear2 = nn.Linear(256,1)

        self.leaky1 = nn.LeakyReLU()
        self.sig1 = nn.Sigmoid()



    def forward(self, x):

        h = self.linear1(x)
        h = self.leaky1(h)

        h = self.linear2(h)
        h = self.sig1(h)

        return(h)

# TODO 4
# Implement training loop for the classifier:
# for i in range(n0):
#     zero gradients
#     calculate predictions for given x
#     calculate loss, comparing the predictions with the given y
#     calculate the gradient (loss.backward())
#     print i and the loss
#     perform an optimizer step
def train_classifier(opt, model, x, y):

    optimizer = opt

    #x = 0
    #y = 1

    for i in range(n0):
      optimizer.zero_grad()

      y_pred= model(x)
      loss = loss_fn(y_pred, y)
      print(loss)
      loss.backward()


      optimizer.step()
    pass

# TODO 5
# Instantiate the network and the optimizer
# call train_classifier with the training set
# Calculate metrics on the validation set
# Example:
#      y_pred = net(x_validation[labels_validation == 3]) calculates all predictions for all images we know to be 3s
#      (y_pred > 0.5) is a tensor that tells you if a given image was classified as your chosen digit (True) or not (False)
#      You can convert this tensor to 0s and 1s by calling .float()
#      (y_pred > 0.5).sum() will tell you how many of these predictions were true
# You are supposed to calculate:
#     For each digit from 0 to 9, which number percentage of images that were of that digit were predicted as your chosen digit
#     The percentage of digits that were classified correctly (i.e. that were your digit and predicted as such, or were another digit and not predicted as your digit)
#     This last value (accuracy) should be over 90% (preferably over 98%; precision and recall may be lower than that, 90-93% would be decent values)
#     Precision (which percentage of images identified as your chosen digit was actually that digit: TP/(TP+FP))
#     Recall (which percentage of your chosen digit was identified as such: TP/(TP+FN))
def classify(x_train, y_train, x_validation, labels_validation):

    model = Discriminator()

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

    train_classifier(optimizer, model, x_train, y_train)

    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    trueNegatives = 0

    for i in range(10):

      y_pred = model(x_validation[labels_validation == i])
      binaryPredictions = (y_pred > 0.5).float()

      truePositives += binaryPredictions.sum().item()
      falsePositives += (binaryPredictions == 1).sum().item()
      falseNegatives += (binaryPredictions == 0).sum().item()
      trueNegatives += ((binaryPredictions == 0) & (labels_validation != i)).sum().item()

    print("True Positives:", truePositives)
    print("False Positives:", falsePositives)
    print("False Negatives:", falseNegatives)
    print("True Negatives:", trueNegatives)
    print("Accuracy:", (truePositives + trueNegatives) / (truePositives + falsePositives + falseNegatives + trueNegatives))


    pass

"""# GAN (5 TODOs)"""

# TODO 6: Change number of total training iterations for GAN, for the discriminator and for the generator
n = 5
n1 = 5
n2 = 5

# TODO 7
# Change Network architecture of the generator network. It should have 100 inputs (will be random numbers) and 784 outputs (one for each pixel, each between 0 and 1)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 256)
        self.linear2 = nn.Linear(256, 784)
        self.sig = nn.Sigmoid()
        self.leaky = nn.LeakyReLU()
    def forward(self, x):
        h = self.linear1(x)
        print(h)
        h = self.leaky(h)
        h = self.linear2(h)
        return self.sig(h)

# TODO 8
# Implement training loop for the discriminator, given real and fake data:
# for i in range(n1):
#     zero gradients
#     calculate predictions for the x known as real
#     calculate loss, comparing the predictions with a tensor consisting of 1s (we want all of these samples to be classified as real)
#     calculate the gradient (loss_true.backward())
#     calculate predictions for the x known as fake
#     calculate loss, comparing the predictions with a tensor consisting of 0s (we want all of these samples to be classified as fake)
#     calculate the gradient (loss_false.backward())
#     print i and both of the loss values
#     perform an optimizer step
def train_discriminator(opt, discriminator, x_true, x_false):
    print("Training discriminator")
    optimizer = opt
    for i in range(n1):
      optimizer.zero_grad()

      xTruePred = discriminator(x_true)
      loss_true = loss_fn(xTruePred, torch.ones_like(xTruePred))
      loss_true.backward()

      xFalsePred = discriminator(x_false)
      loss_false = loss_fn(xFalsePred, torch.zeros_like(xFalsePred))
      loss_false.backward()

      optimizer.step()

# TODO 9
# Implement training loop for the generator:
# for i in range(n2):
#     zero gradients
#     generate some random inputs
#     calculate generated images by passing these inputs to the generator
#     pass the generated images to the discriminator to predict if they are true or fake
#     calculate the loss, comparing the predictions with a tensor of 1s (the *generator* wants the discriminator to classify its images as real)
#     calculate the gradient (loss.backward())
#     print i and the loss
#     perform an optimization step
def train_generator(opt, generator, discriminator):
    print("Training generator")

    optimizer = opt

    for i in range(n2):
      optimizer.zero_grad()

      generatedImgs = generator(torch.randn(100))

      realOrFake = discriminator(generatedImgs)

      loss = loss_fn(realOrFake, torch.ones_like(realOrFake))

      loss.backward()

      print(loss)
      optimizer.step()

from re import X
# TODO 10
# Implement GAN training loop:
# Generate some random images (with torch.rand) as an initial collection of fakes
# Instantiate the two networks and two optimizers (one for each network!)
# for i in range(n):
#    call train_discriminator with the given real images and the collection of fake images
#    call train_generator
#    generate some images with the current generator, and add a random selection of old fake images (e.g. 100 random old ones, and 100new ones = 200 in total)
#    this will be your new collection of fake images
#    save some of the current fake images to a file (use a filename like "sample_%d_%d.png"%(i,j) so you have some samples from each iteration so you can see if the network improves)
# If you read the todos above, your training code will print the loss in each iteration. The loss for the discriminator and the generator should decrease each time their respective training functions are called
# The images should start to look like numbers after just a few (could be after 1 or 2 already, or 3-10) iterations of *this* loop
def gan(x_real):
    x_false = torch.rand_like(x_real)

    discriminator = Discriminator()
    generator = Generator()

    optDisc = torch.optim.Adam(discriminator.parameters(),lr = 0.01)
    optGenr = torch.optim.Adam(generator.parameters(),lr = 0.01)

    if discriminator:
      print("disc")
    else:
      print("no disk")


    for i in range(10):
        # optDisc.zero_grad()
        # optGenr.zero_grad()

        #def train_discriminator(opt, discriminator, x_true, x_false):

        train_discriminator(optDisc, discriminator, x_real, x_false.detach())


        #def train_generator(opt, generator, discriminator):

        train_generator(optGenr, generator, discriminator)

        new_images = generator(torch.randn(256,100))

        x_false = torch.cat([x_false[:100], new_images], dim=0)


        # Save some of the current fake images to a file
        for j in range(200):  # Save 5 images in each iteration
          show_image(new_images[j], f"sample_{i}_{j}.png", scale=SCALE_01)

"""# Main (no TODOs)"""

def main(rungan):
    """
    You do not have to change this function!

    It will:
        automatically download the data set if it doesn't exist yet
        make sure all tensor shapes are correct
        normalize the images (all pixels between 0 and 1)
        provide labels for the classification task (0 for all images that are not your digit, 1 for the ones that are)
        extract the images of your chosen digit for the GAN
    """
    train = torchvision.datasets.MNIST(".", download=True)
    x_train = train.data.float().view(-1,28*28)/255.0
    labels_train = train.targets
    y_train = (labels_train == digit).float().view(-1,1)

    validation = torchvision.datasets.MNIST(".", train=False)
    x_validation = validation.data.float().view(-1,28*28)/255.0
    labels_validation = validation.targets

    if rungan:
        gan(x_train[labels_train == digit])
    else:
        classify(x_train, y_train, x_validation, labels_validation)

"""# Test call (TODO: TEST)"""

# NOTE: This will not work until you have done TODO 1 above!
# If you have not done TODO 1 yet, you will get: AttributeError: 'bool' object has no attribute 'float'
GAN = True
main(GAN)