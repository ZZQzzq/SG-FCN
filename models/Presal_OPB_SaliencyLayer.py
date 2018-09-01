# -*- coding: utf-8 -*-
# File  : ContinuFrameDataLayer.py
# Author: ZZQ
# Date  : 18-06-14

import random
import numpy as np
from PIL import Image
import scipy.io
import cv2
import sys
sys.path.append("/yourpathtocaffe/caffe/python")
sys.path.append("/yourpathtocaffe/caffe/caffe")
import caffe


class SaliencyDetDataLayer(caffe.Layer):
    """
    Load (input video, label video) pairs 
    one-at-a-time while reshaping the net to preserve dimensions.

    This data layer has four tops:

    1. the data 
	2. the pre saliency map
	3. the salient object boundary map
    4. the label

    Use this to feed data to the sgf serious.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - video_dir: path to video
        - action: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for video saliency detection[].

        example: params = dict(video_dir="/path/to/video", split="val")
        """
        # config
        params = eval(self.param_str)
        self.video_dir = params['video_dir']
        self.split = params['action']
        self.mean = np.array((104.00699, 116.66877, 122.67892), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # four tops: 
        if len(top) != 4:
            raise Exception("Need to define four tops: current frame, presal, bmap and ground truth.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for frames and labels
        split_f = '{}/{}.txt'.format(self.video_dir, self.split)  # get txt file for train/val
        self.indices = open(split_f, 'r').read().splitlines()  # get txt context by line, save in indices
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:  # rank or not (train:1, test:0)
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices) - 1)

    def reshape(self, bottom, top):

        # load 'image + presal + bmap + label' pair
        self.frame = self.load_image(self.indices[self.idx] + '/2')  # frame
        self.presal = self.load_presal(self.indices[self.idx])  # presal 
        self.bmap = self.load_bmap(self.indices[self.idx])  # object boundary salmap 
        self.gt = self.load_label(self.indices[self.idx] + '/2', 'gt2', label_type='saliency')  # gt data

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.frame.shape)
        top[1].reshape(1, *self.presal.shape)
        top[2].reshape(1, *self.bmap.shape)
        top[3].reshape(1, *self.gt.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.frame
        top[1].data[...] = self.presal
        top[2].data[...] = self.bmap
        top[3].data[...] = self.gt

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices) - 1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/VIDEO/{}.jpg'.format(self.video_dir, idx))
        out = im.resize((500, 500))
        in_ = np.array(out, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_
    
    def load_presal(self, idx):
        """
        Load presal
        """
        im = Image.open('{}/PRESAL/{}.png'.format(self.video_dir, idx))
        out = im.resize((500, 500))
        in_ = np.array(out, dtype=np.float32)
        in_ = in_[np.newaxis, ...]
        return in_

    def load_bmap(self, idx):
        """
        Load boundary object salmap
        """
        im = Image.open('{}/BMAP/{}/{}.png'.format(self.video_dir, idx, idx))
        out = im.resize((500, 500))
        in_ = np.array(out, dtype=np.float32)
        if (np.max(in_) - np.min(in_)) > 0:
            in_ = (in_ - np.min(in_)) / (np.max(in_) - np.min(in_))
        in_ = in_[np.newaxis, ...]
        return in_

    def load_label(self, idx, name, label_type=None):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        if label_type == 'saliency':
            label = scipy.io.loadmat('{}/GT/{}.mat'.format(self.video_dir, idx))[name]
            label = cv2.resize(label, dsize=(500, 500))
        else:
            raise Exception("Unknown label type: {}. Pick saliency.".format(label_type))
        label = label[np.newaxis, ...]
        return label.copy()
