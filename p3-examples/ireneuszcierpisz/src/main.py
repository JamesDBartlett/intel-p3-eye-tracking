import cv2
import numpy as np
from face_detection import Model_Face
from gaze_estimation import Model_Gaze
from head_pose_estimation import Model_HeadPose
from facial_landmarks_detection import Model_Landmarks
from mouse_controller import MouseController
from input_feeder import InputFeeder
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)

demo = 'bin/demo.mp4'
face = 'bin/FACE0.png'

face_detect = "models/face-detection-adas-binary-0001"
gaze_estim = "models/gaze-estimation-adas-0002"
head_pose = "models/head-pose-estimation-adas-0001"
landmarks = "models/landmarks-regression-retail-0009"


def main(args):
    device = args.device
    
    precision, speed = args.mouse_precision, args.mouse_speed
    mouse = MouseController(precision=precision, speed=speed)    
    
    #get paths to the models
    modelF, modelG, modelH, modelL = args.modelF, args.modelG, args.modelH, args.modelL
    
    face = Model_Face(modelF, device)
    gaze = Model_Gaze(modelG, device) 
    headpose = Model_HeadPose(modelH, device)    
    landmarks = Model_Landmarks(modelL, device)
    
    face.load_model()
    gaze.load_model()
    headpose.load_model()
    landmarks.load_model()
    
    input_type, input_file = args.input_type, args.input_file
    feed = InputFeeder(input_type=input_type, input_file=input_file)
    vframe_shape = feed.load_data()

    logging.info("Please wait. Processing inference...")
    # Run inference on four models and get outputs
    for batch in feed.next_batch():
        frame_copy = batch.copy()
##face:
        frame4infer_f = face.preprocess_input(batch)
        face_output = face.predict(frame4infer_f)
        #get face bb coordinates:
        f_preprocessed_output = face.preprocess_output(face_output, vframe_shape) 
        xmin,ymin,xmax,ymax = f_preprocessed_output
##headpose:
        frame4infer_h = headpose.preprocess_input(batch)
        # get yaw, pitch and roll head pose angles
        headpose_output = headpose.predict(frame4infer_h)
        h_preprocessed_output = headpose.preprocess_output(headpose_output)
##landmarks:
        #get roi of face
        roi = batch[ymin:ymax, xmin:xmax] 
        frame4infer_l = landmarks.preprocess_input(roi)        
        landmarks_output = landmarks.predict(frame4infer_l)
        # get landmarks coordinates
        l_preprocessed_output = landmarks.preprocess_output(landmarks_output, f_preprocessed_output)
        right_eye, left_eye, nose, right_lip_corner, left_lip_corner = l_preprocessed_output
##gaze        
        r_eye_crop = batch[right_eye[1]-20:right_eye[1]+20, right_eye[0]-20:right_eye[0]+20]
        l_eye_crop = batch[left_eye[1]-20:left_eye[1]+20, left_eye[0]-20:left_eye[0]+20]
        re_blob4infer_g = gaze.preprocess_input(r_eye_crop, 're')
        le_blob4infer_g = gaze.preprocess_input(l_eye_crop, 'le')
        hp_blob4infer_g = gaze.preprocess_input(np.array(h_preprocessed_output), 'hp')        
        gaze_output = gaze.predict(re_blob4infer_g, le_blob4infer_g, hp_blob4infer_g)
        g_preprocessed_output = gaze.preprocess_output(gaze_output, l_preprocessed_output, vframe_shape)
        
        # Get mouse pointer position
        x, y = g_preprocessed_output
          
        # Move a mouse pointer
        mouse.move(x, y)

        if input_type == "image":
            cv2.imwrite("output_image.jpg", frame_copy)
            logging.info("        ! Got output image!")
        
        if input_type == 'video':
            feed.write(frame_copy)
            
        
    feed.close()
    logging.info("End of the processing.")
    
if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--modelF', type=str, help="The location of the model XML file", default=face_detect) 
    parser.add_argument('--modelG', default=gaze_estim) 
    parser.add_argument('--modelH', default=head_pose) 
    parser.add_argument('--modelL', default=landmarks)     
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--input_type', type=str, help="'video', 'image' or 'cam'", default='video')    
    parser.add_argument('--input_file', help="The location of the video or image file. 'None' for input type: 'cam'", default=demo)  
    parser.add_argument('--mouse_speed', default='medium')  
    parser.add_argument('--mouse_precision', default='medium')     
    
    args=parser.parse_args()
    main(args)