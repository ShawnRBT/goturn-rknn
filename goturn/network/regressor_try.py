# Date: Friday 02 June 2017 05:04:00 PM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Basic regressor function implemented

from __future__ import print_function
import os
import glob
import numpy as np
import sys
import cv2
from helper import config
from rknn.api import RKNN

# sys.path.insert(0, config.CAFFE_PATH)


class regressor:
    """Regressor Class"""

    def __init__(self, deploy_proto, rknn_model, gpu_id, num_inputs,
                 do_train, logger, solver_file=None):
        """TODO: to be defined"""

        self.num_inputs = num_inputs
        self.logger = logger
        self.rknn_model = rknn_model
        self.modified_params_ = False
        self.mean = [104, 117, 123]
        self.modified_params = False
        self.solver_file = None
        self.height = 227
        self.width = 227
        self.channels = 3

        if solver_file:
            self.solver_file = solver_file

    def preprocess(self, image):
        """TODO: Docstring for preprocess.

        :arg1: TODO
        :returns: TODO

        """
        num_channels = self.channels
        if num_channels == 1 and image.shape[2] == 3:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif num_channels == 1 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif num_channels == 3 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif num_channels == 3 and image.shape[2] == 1:
            image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_out = image

        if image_out.shape != (self.height, self.width, self.channels):
            image_out = cv2.resize(image_out, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        image_out = np.float32(image_out)
        image_out -= np.array(self.mean)
        image_out = np.transpose(image_out, [2, 0, 1])
        return image_out

    def regress(self, curr_search_region, target_region, rknn):
        """TODO: Docstring for regress.
        :returns: TODO

        """
        print("estimate")
        return self.estimate(curr_search_region, target_region, rknn)

    def estimate(self, curr_search_region, target_region, rknn):
        """TODO: Docstring for estimate.

        :arg1: TODO
        :returns: TODO

        """
        # net = self.net
        #
        # # reshape the inputs
        #
        # net.blobs['image'].data.reshape(1, self.channels, self.height, self.width)
        # net.blobs['target'].data.reshape(1, self.channels, self.height, self.width)
        # net.blobs['bbox'].data.reshape(1, 4, 1, 1)
        #
        curr_search_region = self.preprocess(curr_search_region)
        target_region = self.preprocess(target_region)
        #
        # net.blobs['image'].data[...] = curr_search_region
        # net.blobs['target'].data[...] = target_region
        # net.forward()
        # bbox_estimate = net.blobs['fc8'].data


        
        # set inputs
        # curr_search_region = np.reshape(curr_search_region, (1, 3, 227, 227))
        # target_region = np.reshape(target_region, (1, 3, 227, 227))
        # input = curr_search_region, target_region

        curr_search_region = np.reshape(curr_search_region, (1, 3, 227, 227))
        target_region = np.reshape(target_region, (1, 3, 227, 227))
        print("curr", curr_search_region.shape)
        print("tar", target_region.shape)
        bbox = np.ones(shape = (1, 4, 1, 1), dtype = np.int8)


        # Inference
        print('--> Running model')
        [output] = rknn.inference(inputs=[curr_search_region, target_region, bbox])
        print("bbox_estimate =", output)
        print("(0, 0) =", output[0, 0])
        bbox_estimate = output
        print('done')

        return bbox_estimate