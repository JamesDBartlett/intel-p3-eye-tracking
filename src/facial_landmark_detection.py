"""
    facial_landmark_detection.py
    Author: @JamesDBartlett3
"""
import cv2
import time
import numpy as np
from inference import Network


class FacialLandmarkDetection:
    """
        Facial Landmark Detection Class
    """

    def __init__(self, model_name, device="CPU", extensions = None):
        """
            set instance variables
        """
        self.model_xml = model_name
        self.device =  device
        self.extensions = extensions
        self.infer_network = Network()

    def load_model(self):
        """
            load the model specified by the user
        """
        self.infer_network.load_model(self.model_xml, self.device, self.extensions)

    def predict(self, image):
        """
            run predictions on the input image
        """
        self.infer_network.exec_net(image)
        if(self.infer_network.wait() == 0):
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

    def preprocess_output(self, outputs, box, img, overlay_inference, probability_threshold):
        """
            preprocess output image
        """
        landmarks = outputs.reshape(1,10)[0]
        h, w = (box[3] - box[1], box[2] - box[0])
        if(overlay_inference):
            for i in range(2):
                x, y = (w * int(landmarks[i*2]), h * int(landmarks[i*2+1])
                cv2.circle(img, (box[0]+x, box[1]+y), 25, (i*255, 255, 0), 2)
        lp = [w * landmarks[0], h * landmarks[1]]
        rp = [w * landmarks[2], h * landmarks[3]]
        return img, lp, rp