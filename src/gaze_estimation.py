"""
    gaze_estimation.py
    Author: @JamesDBartlett3
"""
import cv2
import time
import numpy as np
from inference import Network
from utility import color, axes_misc


def print_coords(img, axis_name, axis_value, color):
    cv2.putText(
        img,
        axis_name + str("{:.1f}".format(axis_value * axes_misc[axis_name][1])),
        (15, axes_misc[axis_name][0]),
        0,
        color,
        1,
    )


def xy_min_max(p, d, m, f):
    return int(p + (d * m) // 2) if int(p + (d * m) // 2) >= f else 0


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

    def preprocess_input(self, frame, face, l_coords, r_coords, overlay_inference):
        """
            preprocess input image
        """
        l_shape = r_shape = [1, 3, 50, 50]
        l_image, r_image, l_frame, r_frame = [None] * 4
        L, R = (
            [l_coords, l_shape, l_image, 20, l_frame],
            [r_coords, r_shape, r_image, 100, r_frame],
        )
        p_frames = []
        for i in (L, R):
            x = i[0][0]
            y = i[0][1]
            h = i[1][2]
            w = i[1][3]
            xmin = xy_min_max(x, w, -1, 0)
            xmax = xy_min_max(x, w, 1, face.shape[1])
            ymin = xy_min_max(y, h, -1, 0)
            ymax = xy_min_max(y, h, 1, face.shape[0])
            i[2] = face[ymin:ymax, xmin:xmax]
            if overlay_inference:
                frame[150 : 150 + i[2].shape[0], i[3] : i[3] + i[2].shape[1]] = i[2]
            i[4] = cv2.resize(i[2], (w, h))
            i[4] = i[4].transpose((2, 0, 1))
            i[4] = i[4].reshape(1, i[4].shape)
            p_frames.append(i[4])
        return frame, p_frames[0], p_frames[1]

    def preprocess_output(
        self,
        outputs,
        img,
        box,
        l_coords,
        r_coords,
        overlay_inference,
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
                        print_coords(
                            img, axes_misc[j], axis, color[k]
                        )
                cv2.arrowedLine(
                    img,
                    (lx, ly),
                    (lx + int(x * 100), ly + int(-100 * y)),
                    color.values[2],
                    5,
                )
                cv2.arrowedLine(
                    img,
                    (rx, ry),
                    (rx + int(x * 100), ry + int(-100 * y)),
                    color.values[2],
                    5,
                )
        return img, [x, y, z]
