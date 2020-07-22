"""
    main.py
    Author: @JamesDBartlett3
"""

# Import official libraries
import os
import cv2
import sys
import time
import random
import numpy as np
import logging
from argparse import ArgumentParser

# Import Udacity-provided libraries
from mouse_controller import MouseController
from input_feeder import InputFeeder

# Import libraries I wrote
from face_detection import FaceDetection
from facial_landmark_detection import FacialLandmarkDetection
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation


def setup_argparser():
    argparser = ArgumentParser()
    argparser.add_argument("-fd", "--face_detection", required = True, type = str,
                            help = "Path to Face Detection model .xml file")
    argparser.add_argument("-fld", "--facial_landmark_detection", required = True, type = str,
                            help = "Path to Facial Landmark Detection model .xml file")
    argparser.add_argument("-ge", "--gaze_estimation", required = True, type = str,
                            help = "Path to Gaze Estimation model .xml file")
    argparser.add_argument("-hpe", "--head_pose_estimation", required = True, type = str,
                            help = "Path to Head Pose Estimation model .xml file")
    argparser.add_argument("-i", "--input", required = False, type = str,
                            help = "Path to input video file, or 'webcam' for webcam feed",
                            default = "../media/demo.mp4")
    argparser.add_argument("-d", "--device", required = False, type = str, default = "CPU",
                            help = "Target device: CPU (default), GPU, FPGA, or MYRIAD")
    argparser.add_argument("-c", "--cpu_extension", required = False, type = str, default = None,
                            help = "Absolute path to CPU extension library file (optional)")
    argparser.add_argument("-p", "--probability_threshold", required = False, type = float,
                            default = 0.6, help = "Threshold of model confidence, below which "
                                                    "detections will not be counted (default: 0.6)")
    argparser.add_argument("-l", "--logfile", type = str, default = None, required = False,
                            help = "Specify logfile name. Default behavior is no logging.")
    argparser.add_argument("-oi", "--overlay_inference", default = False, required = False,
                            help = "Overlay inference output on video", action = "store_true")
    argparser.add_argument("-mc", "--mouse_control", default = False, required = False,
                            help = "Allow application to control mouse cursor", action = "store_true")
    argparser.add_argument("-vw", "--video_window", default = False, required = False,
                            help = "Live stream output video in a window", action = "store_true")
    return argparser


def run_inference(args):
    """
        Take args input from main, run inference on input video, and display/save output video
    """

    def run(args):
        print(args)

    
    if isinstance(args.logfile, str):
        print("Logfile: " + args.logfile)
        try:
            logging.basicConfig(
                level = logging.INFO,
                format = "%(levelname)s :: %(message)s (%(asctime)s)",
                handlers = [logging.FileHandler(args.logfile), logging.StreamHandler()]
            )
            run(args)
        except Exception as e:
            logging.exception("Error:" + str(e))
            # print("Logfile: " + args.logfile)

    else:
        print("Logging disabled.")
        run(args)





def main():
    """
        Get args from input & pass them to run_inference()
    """
    args = setup_argparser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
