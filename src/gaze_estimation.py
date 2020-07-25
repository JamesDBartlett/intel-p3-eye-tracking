"""
    gaze_estimation.py
    Author: @JamesDBartlett3
"""
import cv2
import time
import numpy as np
from inference import Network
from utility import color, axes_misc

def print_coords(img, axis_name, axis_value, color, probability_threshold):
    cv2.putText(
                img,
                axis_name + str("{:.1f}".format(axis_value * axes_misc[axis_name][1])),
                (15, axes_misc[axis_name][0]),
                0,
                probability_threshold,
                color,
                1,
            )

class GazeEstimation:
    """
        Gaze Estimation Class
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

    def preprocess_input(self, frame, box, l_coords, r_coords, overlay_inference):
        """
            preprocess input image
        """
        l_shape = r_shape = [1, 3, 60, 60]

    def preprocess_output(
        self,
        outputs,
        img,
        box,
        l_coords,
        r_coords,
        overlay_inference,
        probability_threshold,
    ):
        """
            preprocess output image
        """
        x, y, z = [outputs[0][i] for i in range(len(outputs[0]))]

        if overlay_inference:
            for axis in [x, y, z]:
                for j in axes_misc.keys():
                    for k in color.keys():
                        print_coords(img, axes_misc[j], axis, color[k], probability_threshold)


