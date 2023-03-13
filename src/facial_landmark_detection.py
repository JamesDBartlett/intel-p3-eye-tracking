"""
    facial_landmark_detection.py
    Author: @JamesDBartlett3@techhub.social
"""
import cv2
import time
import numpy as np
from inference import Network


class FacialLandmarkDetection:
    """
        Facial Landmark Detection Class
    """

    def __init__(self, model, device="CPU", extensions=None):
        """
            set instance variables
        """
        self.model_xml = model
        self.device = device
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
        return (
            self.infer_network.get_output()[self.infer_network.output_blob]
            if self.infer_network.wait() == 0
            else None
        )

    def preprocess_input(self, image):
        """
            preprocess input image
        """
        input_shape = self.infer_network.get_input_shape()
        frame = np.copy(image)
        frame = cv2.resize(frame, (input_shape[3], input_shape[2])).transpose((2, 0, 1))
        return frame.reshape(1, *frame.shape)

    def preprocess_output(self, outputs, box, img, overlay_inference):
        """
            preprocess output image
        """
        landmarks = outputs.reshape(1, 10)[0]
        h, w = (box[3] - box[1], box[2] - box[0])

        # This is broken, but I can't figure out why...
        overlay_inference = False # ...so I've disabled it for now.
        if overlay_inference:
            for e in range(2):
                x, y = (w * int(landmarks[e * 2]), h * int(landmarks[e * 2 + 1]))
                cv2.circle(img, (box[0] + x, box[1] + y), 30, (0, 255, e * 255), 2)

        return (
            img,
            [w * landmarks[0], h * landmarks[1]],
            [w * landmarks[2], h * landmarks[3]],
        )

