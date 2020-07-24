"""
    head_pose_estimation.py
    Author: @JamesDBartlett3
"""
import cv2
import time
import numpy as np
from inference import Network


class HeadPoseEstimation:
    """
        Head Pose Estimation Class
    """

    def __init__(self, model_name, device="CPU", extensions = None):
        """
            set instance variables
        """
        raise NotImplementedError

    def load_model(self):
        """
            loading the model specified by the user to the device
        """
        raise NotImplementedError

    def predict(self, image):
        """
            run predictions on the input image
        """
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        """
            preprocess input image
        """
        raise NotImplementedError

    def preprocess_output(self, outputs):
        """
            preprocess output image
        """
        raise NotImplementedError
