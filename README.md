# Triplet Loss & Batch triplet loss & SVM
Here, I have some experiments in order to make more accurate classifier. Triplet Loss, Batch Triplet, SVM are used for experiments. softmax features + SVM was the best in my result as a record of 91% top1 accuracy.

## Dataset

[Kaggle's State Farm Distracted Driver Detection Dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection) is used as traning dataset, you should modify `sampledata.py` to fit your dataset.

## Setup

Rebuild your caffe directory:

	cd $CAFFEROOT$
	cp Makefile.configexample Makefile.config

Remember to uncomment the line to makesure your python layers could be found:

	WITH_PYTHON_LAYER := 1

Then build caffe and pycaffe:

	make all -j8 & make pycaffe

## Usage

1. Modify `sampledata.py`, `config.py` and `train.py` to fit your dataset and working environment.

2. Pre-train your model with softmax loss.

3. Finetune triplet model based on your pre-trained model.

4. Learn to adjust parameters.

## Reference
This project is based on https://github.com/hizhangp/triplet
