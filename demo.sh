# Run this script to execute the demo

python3 src/main.py \
        -fd models/intel/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml \
        -fld models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
        -ge models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml \
        -hpe models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \
        -i media/demo.mp4 \
        -d HETERO:MYRIAD,CPU \
        --print 
        --no_move
        