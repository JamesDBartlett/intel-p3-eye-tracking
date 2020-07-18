python3 src/main.py -fd intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
    -lr intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml \
    -hp intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml \
    -ge intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml \
    -i bin/demo.mp4 -flags ff fl fh fg