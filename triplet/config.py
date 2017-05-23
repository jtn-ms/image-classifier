import os

# Training image data path
IMAGEPATH = '/media/frank/Data/Database/ImageNet/Kaggle/train/'

# Mean.binaryproto data path
MEAN_PROTO = '/home/frank/triplet-master/data/models/softmax/mean.binaryproto'
MEAN_NPY = '/home/frank/triplet-master/data/models/softmax/mean.npy'
MEAN_JPG = '/home/frank/digits/digits/jobs/20170512-093826-326b/mean.jpg'

# LFW image data path
LFW_IMAGEPATH = 'data/LFW/lfw-deepfunneled/'

# Path to caffe directory
CAFFEPATH = '/home/frank/caffe-segnet'

# Channel Size(BGR:3 or Gray:1)
CHANNEL_SIZE = 1

# Image Size
IMAGE_SIZE = 227

# Snapshot iteration
SNAPSHOT_ITERS = 132

# Max training iteration
MAX_ITERS = 3960

# The number of samples in each minibatch for triplet loss
TRIPLET_BATCH_SIZE = 20

# The number of samples in each minibatch for other loss
BATCH_SIZE = 20

# If need to train tripletloss, set False when pre-train
TRIPLET_LOSS = True

# Use horizontally-flipped images during training?
FLIPPED = False

# training percentage
PERCENT = 0.75

# USE semi-hard negative mining during training?
SEMI_HARD = True

# Number of samples of each identity in a minibatch
CUT_SIZE = 5
