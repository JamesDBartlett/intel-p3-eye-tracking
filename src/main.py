"""
    main.py
    Author: @JamesDBartlett3
"""

# Import official libraries
import os
import cv2
import sys
import time
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
        "-fl",
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
        "-hp",
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
        "-ce",
        "--cpu_extension",
        required=False,
        type=str,
        default=None,
        help="Absolute path to CPU extension library file (optional)",
    )
    argparser.add_argument(
        "-pt",
        "--probability_threshold",
        required=False,
        type=float,
        default=0.6,
        help="Threshold of model confidence, below which "
        "detections will not be counted (default: 0.6)",
    )
    argparser.add_argument(
        "-lf",
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

mouse_control = MouseController("low", "fast")

def now():
    return time.time()


def log_model_load_times(logging_enabled, load_start, fl_start, ge_start, hp_start):
    """
        if logging is enabled, log the model loading times
    """
    if logging_enabled:
        logging.info("--------------------------------------------")
        logging.info("========= Model Loading Times (ms) =========")
        logging.info("Face Detection: {:.2f}".format((now() - load_start) / 0.001))
        logging.info(
            "Facial Landmark Detection: {:.2f}".format((now() - fl_start) / 0.001)
        )
        logging.info("Gaze Estimation: {:.2f}".format((now() - ge_start) / 0.001))
        logging.info("Head Pose Estimation: {:.2f}".format((now() - hp_start) / 0.001))
        logging.info("____________________________________________")
        logging.info(
            "Total Loading Time (All Models): {:.2f}".format(
                (now() - load_start) / 0.001
            )
        )
        logging.info("============================================")
        logging.info("                                            ")


def log_inference_times(
    logging_enabled, frame_count, fd_time, fl_time, ge_time, hp_time
):
    """
        if logging is enabled and frame count is greater than 0, log the inference times
    """
    if logging_enabled and frame_count > 0:
        logging.info("----------------------------------------------")
        logging.info("========= Model Inference Times (ms) =========")
        logging.info("Face Detection: {:.2f}".format((fd_time / frame_count) / 0.001))
        logging.info(
            "Facial Landmark Detection: {:.2f}".format((fl_time / frame_count) / 0.001)
        )
        logging.info("Gaze Estimation: {:.2f}".format((ge_time / frame_count) / 0.001))
        logging.info(
            "Head Pose Estimation: {:.2f}".format((hp_time / frame_count) / 0.001)
        )
        logging.info("============================================")


def user_quit(logging_enabled):
    """
        if logging is enabled, log the user_quit message.
    """
    if logging_enabled:
        logging.info("User pressed Esc or Q key. Quitting...")


def infer(args, logging_enabled):
    """
        run inference on input video, display/save output video
    """
    face_detection = FaceDetection(args.face_detection)
    facial_landmark_detection = FacialLandmarkDetection(args.facial_landmark_detection)
    gaze_estimation = GazeEstimation(args.gaze_estimation)
    head_pose_estimation = HeadPoseEstimation(args.head_pose_estimation)
    load_start = now()
    face_detection.load_model()
    fl_start = now()
    facial_landmark_detection.load_model()
    ge_start = now()
    gaze_estimation.load_model()
    hp_start = now()
    head_pose_estimation.load_model()
    log_model_load_times(logging_enabled, load_start, fl_start, ge_start, hp_start)
    feeder = InputFeeder("video", args.input)
    feeder.load_data()
    frame_count, fd_time, fl_time, ge_time, hp_time = [0] * 5
    while 1:
        key = cv2.waitKey(20)
        try:
            frame = next(feeder.next_batch())
        except StopIteration:
            break
        frame_count += 1
        fd_frame = face_detection.preprocess_input(frame)
        inf_start = now()
        fd_output = face_detection.predict(fd_frame)
        fd_time += now() - inf_start
        out_frame, faces = face_detection.preprocess_output(
            fd_output, frame, args.overlay_inference, args.probability_threshold
        )
        detected_face = frame[faces[0][1] : faces[0][3], faces[0][0] : faces[0][2]]
        fl_frame = facial_landmark_detection.preprocess_input(detected_face)
        fl_start = now()
        fl_output = facial_landmark_detection.predict(fl_frame)
        fl_time += now() - fl_start
        out_frame, l_coord, r_coord, = facial_landmark_detection.preprocess_output(
            fl_output, faces[0], out_frame, args.overlay_inference
        )
        hp_frame = head_pose_estimation.preprocess_input(detected_face)
        hp_start = now()
        hp_output = head_pose_estimation.predict(hp_frame)
        hp_time += now() - hp_start
        out_frame, head_pose = head_pose_estimation.preprocess_output(
            hp_output, out_frame, detected_face, faces[0], args.overlay_inference
        )
        out_frame, l_eye, r_eye = gaze_estimation.preprocess_input(
            out_frame, detected_face, l_coord, r_coord, args.overlay_inference
        )
        ge_start = now()
        ge_output = gaze_estimation.predict(head_pose, l_eye, r_eye)
        ge_time += now() - ge_start
        out_frame, g_vec = gaze_estimation.preprocess_output(
            ge_output, out_frame, faces[0], l_coord, r_coord, args.overlay_inference
        )
        if args.video_window:
            cv2.imshow(
                "Computer-Human Interface Peripheral Signal Manipulation via AI Retina Tracking (CHIPSMART)",
                out_frame,
            )
        if args.mouse_control and frame_count % 12 == 0:
            mouse_control.move(g_vec[0], g_vec[1])
        # Quit if user presses Esc or Q
        if key in (27, 81):
            user_quit(logging_enabled)
            break
    log_inference_times(
        logging_enabled, frame_count, fd_time, fl_time, ge_time, hp_time
    )
    feeder.close()
    cv2.destroyAllWindows()
    quit()


def run_inference(args):
    """
        Take args input from main, enable/disable logging, pass args to infer()
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
        handlers=[logging.FileHandler(args.logfile), logging.StreamHandler()],
    )
    if args.mouse_control:
        mouse_control.center()

    if len(args.logfile) > 0:
        print("Logfile: " + args.logfile)
        try:
            infer(args, logging_enabled=True)
        except Exception as e:
            logging.exception(str(e))
    else:
        print("Logging disabled. To enable logging, use the '--logfile' argument.")
        infer(args, logging_enabled=False)


def main():
    """
        Get args from input & pass them to run_inference()
    """
    args = setup_argparser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
