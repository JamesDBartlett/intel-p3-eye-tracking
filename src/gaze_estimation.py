"""
    gaze_estimation.py
    Author: @JamesDBartlett3
"""
import cv2
import time
import numpy as np
from inference import Network
from utility import color, axes_misc, bgr


def print_coords(img, axis_name, axis_value, color):
    cv2.putText(
        img,
        axis_name + str("{:.2f}".format(axis_value * axes_misc[axis_name][1])),
        (axes_misc[axis_name][0], axes_misc[axis_name][1]+75),
        0,
        0.8,
        color,
        1,
    )


def arrowed_line(img, x, y, qx, qy):
    cv2.arrowedLine(
        img, (qx, qy), (qx + int(x * 100), qy + int(-100 * y)), bgr("M"), 5,
    )


def xy_min_max(p, d, m, f):
    if f == 0:
        return int(p + (d * m) // 2) if int(p + (d * m) // 2) >= f else f
    else:
        return int(p + (d * m) // 2) if int(p + (d * m) // 2) <= f else f


class GazeEstimation:
    """
        Gaze Estimation Class
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

    def predict(self, head_pose, l_image, r_image):
        """
            run predictions on the input image
        """
        self.infer_network.exec_net(head_pose, l_image, r_image)
        return (
            self.infer_network.get_output()[self.infer_network.output_blob]
            if self.infer_network.wait() == 0
            else None
        )

    def preprocess_input(self, frame, face, l_coords, r_coords, overlay_inference):
        """
            preprocess input image
        """
        l_shape, r_shape = [[1, 3, 60, 60]] * 2
        l_image, r_image, l_frame, r_frame = [None] * 4
        eyes = (
            [l_coords, l_shape, l_image, 20, l_frame],
            [r_coords, r_shape, r_image, 100, r_frame],
        )
        p_frames = []
        for i in eyes:
            x = i[0][0]
            y = i[0][1]
            h = i[1][2]
            w = i[1][3]
            xmin = xy_min_max(x, w, -1, 0)
            xmax = xy_min_max(x, w, 1, face.shape[1])
            ymin = xy_min_max(y, h, -1, 0)
            ymax = xy_min_max(y, h, 1, face.shape[0])
            p_image = face[ymin:ymax, xmin:xmax]
            if overlay_inference:
                frame[
                    150 : 150 + p_image.shape[0], i[3] : i[3] + p_image.shape[1]
                ] = p_image
            p_frame = cv2.resize(p_image, (w, h))
            p_frame = p_frame.transpose((2, 0, 1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
            p_frames.append(p_frame)
        return frame, p_frames[0], p_frames[1]

    def preprocess_output(
        self, outputs, img, box, l_coords, r_coords, overlay_inference,
    ):
        """
            preprocess output image
        """
        x, y, z = [outputs[0][i] for i in range(len(outputs[0]))]
        xmin, ymin, _, _, = box
        if overlay_inference:
            lc, rc = (l_coords[0:2], r_coords[0:2])
            lx = int(xmin + lc[0])
            ly = int(ymin + lc[1])
            rx = int(xmin + rc[0])
            ry = int(ymin + rc[1])
            for axis in [x, y, z]:
                for j in axes_misc.keys():
                    for k in color.keys():
                        dummy = k
                        # This is broken, but I can't figure out why:
                        #print_coords(img, j, axis, color[k])
                arrowed_line(img, x, y, lx, ly)
                arrowed_line(img, x, y, rx, ry)
        return img, [x, y, z]
