#model_dir =  "/media/frank/Data/Test/caffe/triplet-master/data/models/softmax/"
#sys.path.insert(0, caffe_root + 'python')
import numpy as np
import sys, time, glob
sys.path.insert(0, '/home/frank/caffe-segnet/python')
import caffe
from caffe.proto import caffe_pb2
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn import svm

from skimage import transform as tf
from PIL import Image, ImageDraw
import skimage

subdirs = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
filecounts = [2489,2267,2317,2346,2326,2312,2325,2002,1911,2129]#[10,10,10,10,10,10,10,10,10,10]#

class Recognizer(caffe.Net):
    """
    Recognizer extends Net for image class prediction
    by scaling, center cropping, or oversampling.

    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, mean_file=None,
		 image_dims=(227, 227),
		 raw_scale=255,
                 channel_swap=(2,1,0),
  		 input_scale=None):
        #set GPU mode
        caffe.set_mode_gpu()
        #init net
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean_file is not None:
            proto_data = open(mean_file, "rb").read()
            mean_blob = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
            mean = caffe.io.blobproto_to_array(mean_blob)[0]
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

    def predict(self, input_dir, oversample=True):
        """
        Predict classification probabilities of inputs.

        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.
        oversample : boolean
            average predictions across center, corners, and mirrors
            when True (default). Center-only prediction when False.

        Returns
        -------
        predictions: (N x C) ndarray of class probabilities for N images and C
            classes.
        """
        #load files
        #input_dir='/media/frank/Data/Database/ImageNet/Kaggle/train/c9'
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(input_dir + '/*.jpg')]
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),
                          dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            # Take center crop.
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            crop = crop.astype(int)
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in, 'blobs': ['fc7']})
        predictions = out[self.outputs[0]]
        fc7 = self.blobs['fc7'].data
        print predictions.shape
        print fc7.shape
        # For oversampling, average predictions across crops.
        if oversample:
            predictions = predictions.reshape((len(predictions) / 10, 10, -1))
            predictions = predictions.mean(1)
            #fc7 = fc7.reshape((len(fc7) / 10, 10, -1))
            #fc7 = fc7.mean(1).reshape(-1)
        return predictions, fc7

    def read_imagelist(self,filelist):
        fid=open(filelist)
        lines=fid.readlines()
        test_num=len(lines)
        fid.close()
        X=np.empty((test_num,3,self.image_dims[0],self.image_dims[1]))
        i =0
        for line in lines:
            word=line.split('\n')
            filename=word[0]
            im1=skimage.io.imread(filename,as_grey=False)
            image =skimage.transform.resize(im1,(self.image_dims[0], self.image_dims[1]))*255
            if image.ndim<3:
                print 'gray:'+filename
                X[i,0,:,:]=image[:,:]
                X[i,1,:,:]=image[:,:]
                X[i,2,:,:]=image[:,:]
            else:
                X[i,0,:,:]=image[:,:,2]
                X[i,1,:,:]=image[:,:,0]
                X[i,2,:,:]=image[:,:,1]
            i=i+1
        return X

    def getFeatures(self, X):
        #print "getting features for", file
        out = self.forward_all(data=X)
        #print scores
        fc7 = np.float64(out['deepid'])#self.blobs['fc7'].data   
        return fc7  
             
    def get_features(self, inputdir):
        #print "getting features for", file
        scores,fc7 = self.predict(inputdir)
        return fc7
    
def shuffle_data(features, labels):
	new_features, new_labels = [], []
	index_shuf = range(len(features))
	shuffle(index_shuf)
	for i in index_shuf:
	    new_features.append(features[i])
	    new_labels.append(labels[i])

	return new_features, new_labels

def get_dataset(net):
    features,labels = [],[]
    db_dir = '/media/frank/Data/Database/ImageNet/Kaggle/sam/'#train/'#
    filelist_path = './filelist/'
    ext = '.txt'
    for i in range(len(filecounts)):
        #images = glob.glob(db_dir + subdirs[i] + "/*.jpg")
        #features = features + map(lambda f: get_features(f,net), images)	
        X = net.read_imagelist(filelist_path + subdirs[i] + ext)
        test_num=np.shape(X)[0]
        feature2 = net.getFeatures(X)#net.get_features(db_dir+subdirs[i])#
        #print feature2.shape
        if i == 0:
            features = feature2
        else:
            features = np.concatenate((features, feature2),axis=0)
        labels = labels + [i] * filecounts[i]
    print len(features)
    print len(labels)
    return shuffle_data(features, labels)

model_dir = '/media/frank/Data/Test/caffe/triplet-master/data/models/softmax/'
#net = Recognizer(model_dir + 'deploy.prototxt',
#                 model_dir + 'snapshot_iter_3696.caffemodel',
#                 model_dir + 'mean.binaryproto')
root_dir = '/media/frank/Data/Test/caffe/triplet-master/'
net = Recognizer(root_dir + 'models/deploy.prototxt',
                 root_dir + 'data/models/triplet/alexnet_triplet_iter_600.caffemodel',
                 model_dir + 'mean.binaryproto')
                    
x, y = get_dataset(net)
l = int(len(y) * 0.4)

x_train, y_train = x[: l], y[: l]
x_test, y_test = x[l : ], y[l : ]

clf = svm.SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print "Accuracy: %.3f" % accuracy_score(y_test, y_pred)
