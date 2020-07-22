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
    argparser.add_argument(
        "-fd",
        "--face_detection",
        required=True,
        type=str,
        help="Path to Face Detection model .xml file",
    )
    argparser.add_argument(
        "-fld",
        "--facial_landmark_detection",
        required=True,
        type=str,
        help="Path to Facial Landmark Detection model .xml file",
    )
    argparser.add_argument(
        "-ge",
        "--gaze_estimation",
        required=True,
        type=str,
        help="Path to Gaze Estimation model .xml file",
    )
    argparser.add_argument(
        "-hpe",
        "--head_pose_estimation",
        required=True,
        type=str,
        help="Path to Head Pose Estimation model .xml file",
    )
    argparser.add_argument(
        "-i",
        "--input",
        required=False,
        type=str,
        help="Path to input video file, or 'webcam' for webcam feed",
        default="../media/demo.mp4",
    )
    argparser.add_argument(
        "-d",
        "--device",
        required=False,
        type=str,
        default="CPU",
        help="Target device: CPU (default), GPU, FPGA, or MYRIAD",
    )
    argparser.add_argument(
        "-c",
        "--cpu_extension",
        required=False,
        type=str,
        default=None,
        help="Absolute path to CPU extension library file (optional)",
    )
    argparser.add_argument(
        "-p",
        "--probability_threshold",
        required=False,
        type=float,
        default=0.6,
        help="Threshold of model confidence, below which "
        "detections will not be counted (default: 0.6)",
    )
    argparser.add_argument(
        "-l",
        "--logfile",
        type=str,
        default="",
        required=False,
        help="Specify logfile name. Default behavior is no logging.",
    )
    argparser.add_argument(
        "-oi",
        "--overlay_inference",
        default=False,
        required=False,
        help="Overlay inference output on video",
        action="store_true",
    )
    argparser.add_argument(
        "-mc",
        "--mouse_control",
        default=False,
        required=False,
        help="Allow application to control mouse cursor",
        action="store_true",
    )
    argparser.add_argument(
        "-vw",
        "--video_window",
        default=False,
        required=False,
        help="Live stream output video in a window",
        action="store_true",
    )
    return argparser


def run_inference(args):
    """
        Take args input from main, run inference on input video, and display/save output video
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
        handlers=[logging.FileHandler(args.logfile), logging.StreamHandler()],
    )

    def infer(args):

        def now():
            return time.time()

        mouse_control = MouseController("medium","slow")
        face_detection = FaceDetection(args.face_detection)
        facial_landmark_detection = FacialLandmarkDetection(args.facial_landmark_detection)
        gaze_estimation = GazeEstimation(args.gaze_estimation)
        head_pose_detection = HeadPoseEstimation(args.head_pose_detection)

        logging.info("--------------------------------------------")
        logging.info("========= Model Loading Times (ms) =========")
        start = now()
        face_detection.load_model()
        logging.info("Face Detection: {:.2f}".format((now() - start)) / 0.001)
        start = now()
        facial_landmark_detection.load_model()
        logging.info("Facial Landmark Detection: {:.2f}".format((now() - start)) / 0.001)
        start = now()
        gaze_estimation.load_model()
        logging.info("Gaze Estimation: {:.2f}".format((now() - start)) / 0.001)
        start = now()
        head_pose_detection.load_model()
        logging.info("Head Pose Detection: {:.2f}".format((now() - start)) / 0.001)

        # TODO: Add total load time

        logging.info("--------------------------------------------")
        logging.info("                                            ")

        feeder = InputFeeder("video", args.input)
        feeder.load_data()

        frame_count = 0
        fd_time = 0
        fld_time = 0
        ge_time = 0
        hp_time = 0
        loop = True

        while loop:
            try:
                frame = next(feeder.next_batch())

            except StopIteration:
                break

            key = cv2.waitKey(30)
            frame_count += 1

            preprocessed_frame = face_detection.preprocess_input(frame)
            inf_start = now()
            fd_output = face_detection.predict(preprocessed_frame)
            inf_end = now()
            fd_time += inf_end - inf_start
            out_frame, face_boxes = face_detection.preprocess_output(fd_output, frame, args.overlay_inference)

            for box in face_boxes:
                break
            if key == 27:
                break


    if len(args.logfile) > 0:
        print("Logfile: " + args.logfile)

        logging.info("Nothing to log.")
        try:
            infer(args)
        except Exception as e:
            logging.exception(str(e))
    else:
        print("Logging disabled. To enable logging, use the '--logfile' argument.")
        infer(args)


def main():
    """
        Get args from input & pass them to run_inference()
    """
    args = setup_argparser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
