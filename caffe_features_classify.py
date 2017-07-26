import numpy as np
import sys, time, glob
#caffe_root =  "/home/vagrant/caffe/"
#sys.path.insert(0, caffe_root + 'python')

import caffe
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn import svm

def init_net():
	net = caffe.Classifier(caffe_root  + 'models/bvlc_reference_caffenet/deploy.prototxt',
	                       caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
	net.set_phase_test()
	net.set_mode_cpu()
	# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
	net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
	net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
	return net

def get_features(file, net):
	#print "getting features for", file
	scores = net.predict([caffe.io.load_image(file)])
	feat = net.blobs['fc7'].data[4][:,0, 0]
	return feat

def shuffle_data(features, labels):
	new_features, new_labels = [], []
	index_shuf = range(len(features))
	shuffle(index_shuf)
	for i in index_shuf:
	    new_features.append(features[i])
	    new_labels.append(labels[i])

	return new_features, new_labels

def get_dataset(net, A_DIR, B_DIR):
	CLASS_A_IMAGES = glob.glob(A_DIR + "/*.jpg")
	CLASS_B_IMAGES = glob.glob(B_DIR + "/*.jpg")
	CLASS_A_FEATURES = map(lambda f: get_features(f, net), CLASS_A_IMAGES)
	CLASS_B_FEATURES = map(lambda f: get_features(f, net), CLASS_B_IMAGES)
	features = CLASS_A_FEATURES + CLASS_B_FEATURES
	labels = [0] * len(CLASS_A_FEATURES) + [1] * len(CLASS_B_FEATURES)
	
	return shuffle_data(features, labels)

net = init_net()
x, y = get_dataset(net, sys.argv[1], sys.argv[2])
l = int(len(y) * 0.4)

x_train, y_train = x[: l], y[: l]
x_test, y_test = x[l : ], y[l : ]

clf = svm.SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print "Accuracy: %.3f" % accuracy_score(y_test, y_pred)
