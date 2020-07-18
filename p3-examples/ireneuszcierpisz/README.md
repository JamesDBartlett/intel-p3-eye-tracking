# Computer Pointer Controller
#### The project has been completed as part of the INTEL® EDGE AI FOR IOT DEVELOPERS NANODEGREE by Udacity.

This application is build to run multiple models in the **OpenVino toolkit** on the same machine to **control a computer pointer using eye gaze**.


## Project Set Up and Installation

Folder `src` contains seven files. Four files with model class and its methods. Also `mouse_controller.py`, `input_feeder.py` and main file `main.py`.

There are four OpenVino models in the `models` folder:

   [Face Detection (FP32-INT1)](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
   
   [Head Pose Estimation (FP32)](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
   
   [Facial Landmarks Detection (FP32)](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
   
   [Gaze Estimation (FP32)](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

To build the project I used the **InferenceEngine API from Intel's OpenVino toolkit**.


## Demo

To run the app a basic demo the demo.mp4 file (provided by **Udacity**) can be used, located `bin/demo.mp4` 

Command to run inference on four models:   `python main.py`

The default arguments: --modelF, --modelG, --modelH, --modelL, --input_type "video", --input_file "bin/demo.mp4", --mouse_speed, --mouse_precision, --device "CPU"

Please find below the **result samples** for several different head positions taken from demo.mp4. These examples show how the application works, including tasks performed by subsequent models, until the mouse pointer position is found. 

   [pose_1](https://github.com/ireneuszcierpisz/computer-pointer-controller/blob/master/bin/output_image0.jpg) 
   
   [pose_2](https://github.com/ireneuszcierpisz/computer-pointer-controller/blob/master/bin/output_image1.jpg)  
   
   [pose_3](https://github.com/ireneuszcierpisz/computer-pointer-controller/blob/master/bin/output_image2.jpg) 
   
   [pose_4](https://github.com/ireneuszcierpisz/computer-pointer-controller/blob/master/bin/output_image3.jpg)  
   
   [pose_5](https://github.com/ireneuszcierpisz/computer-pointer-controller/blob/master/bin/output_image6.jpg)


## Documentation

- usage: main.py [-h]   

                        [--modelF MODELF]
                        [--modelG MODELG]
                        [--modelH MODELH]
                        [--modelL MODELL]
                        [--device DEVICE]
                        [--input_type INPUT_TYPE]
                        [--input_file INPUT_FILE]
                        [--mouse_speed MOUSE_SPEED]
                        [--mouse_precision MOUSE_PRECISION]
                        
- optional arguments:

                 -h, --help           show this help message and exit
                 --modelF MODELF      The location of the model XML file
                 --modelG MODELG
                 --modelH MODELH
                 --modelL MODELL
                 --device DEVICE
                 --input_type INPUT_TYPE
                                      'video', 'image' or 'cam'
                 --input_file INPUT_FILE
                                      The location of the video or image file.
                                      'None' for input type: 'cam'
                 --mouse_speed MOUSE_SPEED
                 --mouse_precision MOUSE_PRECISION


## Results

I. Sample of models asynchronous inference performance on a **single image**:

   _face-detection-adas-binary-0001 model precision: FP32-INT1_
   
            1.
            The other three models precision: FP32 
            Performance time: 0.065
            2.
            The other three models precision: FP16 
            Performance time: 0.064
            3.
            The other three models precision: FP32-INT8 
            Inference and process time: 0.063

II. Sample of models asynchronous performance on a **video**:

   Number of frames in the video: 595.0 ,  fps:30.02
   
   _face-detection-adas-binary-0001 model precision: **FP32-INT1**_
   
            1.
            The other three models precision: FP32 
            Performance time: 32.153
            2.
            The other three models precision: FP16 
            Performance time: 33.528
            3.
            The other three models precision: FP32-INT8 
            Performance time: 33.701


### Edge Cases

There will be certain situations that will break inference flow. For instance, multiple people in the frame. 

In that case if multiple faces in the same input frame is detected this app choose one face that which is detected with bigest confidence.
