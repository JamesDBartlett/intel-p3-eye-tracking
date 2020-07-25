"""
    head_pose_estimation.py
    Author: @JamesDBartlett3
"""
import cv2
import time
import math
import numpy as np
from inference import Network


class HeadPoseEstimation:
    """
        Head Pose Estimation Class
    """

    def __init__(self, model_name, device="CPU", extensions=None):
        """
            set instance variables
        """
        self.model_xml = model_name
        self.device = device
        self.extensions = extensions
        self.infer_network = Network()

    def load_model(self):
        """
            loading the model specified by the user to the device
        """
        self.infer_network.load_model(self.model_xml, self.device, self.extensions)

    def predict(self, image):
        """
            run predictions on the input image
        """
        self.infer_network.exec_net(image)
        if self.infer_network.wait() == 0:
            return self.infer_network.get_output()[self.infer_network.out_blob]

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        """
            preprocess input image
        """
        input_shape = self.infer_network.get_input_shape()
        frame = np.copy(image)
        frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape(1, *frame.shape)
        return frame

    def preprocess_output(
        self, outputs, image, face, box, overlay_inference, probability_threshold
    ):
        """
            preprocess output image
        """
        raise NotImplementedError
