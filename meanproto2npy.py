import caffe
import numpy as np
import sys
import triplet.config as cfg

global mean_file
mean_file='/home/frank/triplet-master/data/models/softmax/mean.binaryproto'

if __name__ == '__main__':
    proto_data = open(mean_file, "rb").read()
    mean_blob = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    #mean = caffe.io.blobproto_to_array(mean_blob)[0]
    arr = np.array(caffe.io.blobproto_to_array(mean_blob))
    out = arr[0]
    np.save(cfg.MEAN_NPY,out)
