"""
    face_detection.py
    Author: @JamesDBartlett3@techhub.social
"""
import cv2
import time
import numpy as np
from inference import Network


class FaceDetection:
    """
        Face Detection Class
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
            loading the model specified by the user to the device
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
        frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape(1, *frame.shape)
        return frame

    def preprocess_output(self, outputs, img, overlay_inference, probability_threshold):
        """
            preprocess output image
        """
        h, w = img.shape[0:2]
        boxes = []
        for a in range(len(outputs[0][0])):
            box = outputs[0][0][a]
            confidence = box[2]
            if not confidence <= probability_threshold:
                b = [int(w * box[3]), int(h * box[4]), int(w * box[5]), int(h * box[6])]
                if overlay_inference:
                    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
                boxes.append([b[0], b[1], b[2], b[3]])
        return img, boxes
