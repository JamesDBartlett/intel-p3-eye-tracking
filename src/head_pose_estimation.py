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

        # TODO: REFACTOR THIS CODE

        print(outputs)

        yaw = outputs['angle_y_fc'][0][0]   # Axis of rotation: z
        pitch = outputs['angle_p_fc'][0][0] # Axis of rotation: y
        roll = outputs['angle_r_fc'][0][0] # Axis of rotation: x
        
        #Draw output
        if(overlay_inference):
            cv2.putText(image,"y:{:.1f}".format(yaw), (20,20), 0, 0.6, (255,255,0))
            cv2.putText(image,"p:{:.1f}".format(pitch), (20,40), 0, 0.6, (255,255,0))
            cv2.putText(image,"r:{:.1f}".format(roll), (20,60), 0, 0.6, (255,255,0))
            
            xmin, ymin,_ , _ = box
            face_center = (xmin + face.shape[1] / 2, ymin + face.shape[0] / 2, 0)
            self.draw_axes(image, face_center, yaw, pitch, roll)
        
        return image, [yaw, pitch, roll]

    # code source: https://knowledge.udacity.com/questions/171017
    def draw_axes(self, frame, center_of_face, yaw, pitch, roll):
        focal_length = 950.0
        scale = 50

        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])
        
        # R = np.dot(Rz, Ry, Rx)
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # R = np.dot(Rz, np.dot(Ry, Rx))
        R = Rz @ Ry @ Rx
        # print(R)
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]
        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)
        return frame

    # code source: https://knowledge.udacity.com/questions/171017
    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix
