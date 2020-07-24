"""
    gaze_estimation.py
    Author: @JamesDBartlett3
"""
import cv2
import time
import numpy as np
from inference import Network


class GazeEstimation:
    """
        Gaze Estimation Class
    """

    def __init__(self, model_name, device="CPU", extensions = None):
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

    def preprocess_input(self, frame, face, l_coords, r_coords, overlay_inference):
        """
            preprocess input image
        """
        l_shape = r_shape = [1,3,60,60]
        

    def preprocess_output(self, outputs):
        """
            preprocess output image
        """
        raise NotImplementedError
