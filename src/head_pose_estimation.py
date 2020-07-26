"""
    head_pose_estimation.py
    Author: @JamesDBartlett3
"""
import cv2
import time
import math
import numpy as np
from utility import bgr
from inference import Network


class HeadPoseEstimation:
    """
        Head Pose Estimation Class
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
            self.infer_network.get_output() if self.infer_network.wait() == 0 else None
        )

    def preprocess_input(self, image):
        """
            preprocess input image
        """
        input_shape = self.infer_network.get_input_shape()
        frame = np.copy(image)
        frame = cv2.resize(frame, (input_shape[3], input_shape[2])).transpose((2, 0, 1))
        return frame.reshape(1, *frame.shape)

    def preprocess_output(self, outputs, image, face, box, overlay_inference):
        """
            preprocess output image
        """
        yaw, pitch, roll = (
            outputs["angle_y_fc"][0][0],
            outputs["angle_p_fc"][0][0],
            outputs["angle_r_fc"][0][0],
        )
        x_offset, y_offset = [25] * 2
        if overlay_inference:
            cv2.putText(
                image, "y:{:.1f}".format(yaw), (x_offset, y_offset), 0, 0.6, bgr("G")
            )
            cv2.putText(
                image,
                "p:{:.1f}".format(pitch),
                (x_offset, y_offset * 2),
                0,
                0.6,
                bgr("R"),
            )
            cv2.putText(
                image,
                "r:{:.1f}".format(roll),
                (x_offset, y_offset * 3),
                0,
                0.6,
                bgr("B"),
            )
            xmin, ymin, _, _ = box
            face_center = (xmin + face.shape[1] / 2, ymin + face.shape[0] / 2, 0)
            self.draw_axes(image, face_center, yaw, pitch, roll)
        return image, [yaw, pitch, roll]

    # code source: https://knowledge.udacity.com/questions/171017
    def draw_axes(self, frame, center_of_face, yaw, pitch, roll):
        focal_length = 950.0
        scale = 50
        pi_slice = lambda x: np.pi / 180.0 * 3 * x
        yaw, pitch, roll = list(map(pi_slice, [yaw, pitch, roll]))
        cx, cy = list(map(int, center_of_face[0:2]))
        rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(pitch), -math.sin(pitch)],
                [0, math.sin(pitch), math.cos(pitch)],
            ]
        )
        ry = np.array(
            [
                [math.cos(yaw), 0, -math.sin(yaw)],
                [0, 1, 0],
                [math.sin(yaw), 0, math.cos(yaw)],
            ]
        )
        rz = np.array(
            [
                [math.cos(roll), -math.sin(roll), 0],
                [math.sin(roll), math.cos(roll), 0],
                [0, 0, 1],
            ]
        )

        # Inspired by: learnopencv.com/rotation-matrix-to-euler-angles
        E = rz @ ry @ rx
        camatrix = self.build_camera_matrix(center_of_face, focal_length)
        reshape_arr = lambda arr: np.array((arr), dtype="float32").reshape(3, 1)
        xaxis, yaxis, zaxis, qaxis = list(
            map(
                reshape_arr,
                [
                    [1 * scale, 0, 0],
                    [0, -1 * scale, 0],
                    [0, 0, -1 * scale],
                    [0, 0, 1 * scale],
                ],
            )
        )
        rs = np.array(([0, 0, 0]), dtype="float32").reshape(3, 1)
        c_mat = [camatrix[0][0], camatrix[1][1]]
        rs[2] = c_mat[0]
        euler = lambda axis: np.dot(E, axis) + rs
        a = np.array(list(map(euler, (xaxis, yaxis, zaxis, qaxis))))

        def matmux(i):
            x = int((a[i][0] / a[i][2] * c_mat[0]) + cx)
            y = int((a[i][1] / a[i][2] * c_mat[1]) + cy)
            return (x, y)

        cv2.line(frame, (cx, cy), matmux(0), bgr("R"), 2)
        cv2.line(frame, (cx, cy), matmux(1), bgr("G"), 2)
        cv2.line(frame, matmux(3), matmux(2), bgr("B"), 2)
        cv2.circle(frame, matmux(2), 3, bgr("B"), 2)
        return frame

    # Inspired by: knowledge.udacity.com/questions/171017
    def build_camera_matrix(self, center_of_face, focal_length):
        cx, cy = list(map(int, center_of_face[0:2]))
        camatrix = np.zeros((3, 3), dtype="float32")
        camatrix[0][0], camatrix[1][1] = [focal_length] * 2
        camatrix[0][2] = cx
        camatrix[1][2] = cy
        camatrix[2][2] = 1
        return camatrix
