"""
    main.py
    by @JamesDBartlett3
"""

# Import official libraries
import cv2
import os
import logging
import numpy as np
from argparse import ArgumentParser

# Import Udacity-provided libraries
from mouse_controller import MouseController
from input_feeder import InputFeeder

# Import libraries I wrote
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarkDetection
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
                            help = "Path to input video file, or 'webcam' for webcam feed"
                            default = "../media/demo.mp4")
    argparser.add_argument("-d", "--device", required = False, type = str, default = "CPU"
                            help = "Target device: CPU (default), GPU, FPGA, or MYRIAD")
    argparser.add_argument("-c", "--cpu_extension", required = False, type = str, default = None,
                            help = "Absolute path to CPU extension library file (optional)")
    argparser.add_argument("-p", "--probability_threshold", required = False, type = float,
                            default = 0.6, help = "Threshold of model confidence, below which "
                                                    "detections will not be counted")
    argparser.add_argument("-pf", "--preview_flags", required = False, nargs = '+',
                            default = [], help = "Flags to enable visual output for each of the " 
                                                "models in the stack, space-delimited."
                                                "Valid options: fd fld ge hpe")
    return argparser


def main():
    
    args = setup_argparser().parse_args()
    pFlags = args.preview_flags

    log = logging.getLogger()
    inFile = args.input
    feeder = None
    if not os.path.exists(inFile):
        log.error("Error: Can't find input file")
        exit(1)
    else:
        if inFile.lower() != "webcam":
            feeder = InputFeeder("video", inFile)
        else:
            feeder = InputFeeder("webcam")

    models = {"FaceDetection":args.face_detection, 
            "FacialLandmarkDetection":args.facial_landmark_detection,
            "GazeEstimation":args.gaze_estimation,
            "HeadPoseEstimation":args.head_pose_estimation}

    for m in models.keys():
        if not os.path.exists(models[m]):
            logger.error("Can't find model: " + m + " Please double-check file paths.")
            exit(1)
    
    fd = FaceDetection(m["FaceDetection"], args.device, args.cpu_extension)
    fld = FacialLandmarkDetection(m["FacialLandmarkDetection"], args.device, args.cpu_extension)
    ge = GazeEstimation(m["GazeEstimation"], args.device, args.cpu_extension)
    hpe = HeadPoseEstimation(m["HeadPoseEstimation"], args.device, args.cpu_extension)

    mc = MouseController("medium", "slow")

    feeder.load_data()
    fd.load_model()
    fld.load_model()
    hpe.load_model()
    ge.load_model() 