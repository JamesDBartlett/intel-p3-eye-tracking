# Run this script to execute the demo

python3 src/main.py \
        --face_detection models/intel/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml \
        --facial_landmark_detection models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
        --gaze_estimation models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml \
        --head_pose_estimation models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \
        --input media/demo.mp4 \
        --device CPU \
        --probability_threshold 0.7 \
        --logfile "main.log" \
        --overlay_inference \
        --video_window \
        --mouse_control

# HETERO:MYRIAD,CPU