#!/bin/bash
python3 main.py -fdm intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
    -flm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
    -hpem intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \
    -gem intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml \
    -i bin/demo.mp4 --device "CPU"