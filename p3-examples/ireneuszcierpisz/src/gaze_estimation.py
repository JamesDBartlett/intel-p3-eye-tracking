'''
 The Class for a model. 
'''

from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import logging
logging.getLogger().setLevel(logging.INFO)

class Model_Gaze:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Instance variables:
        '''
        self.model = model_name
        self.device = device
        self.plugin = None
        self.network = None
        self.exec_net = None

    def load_model(self):
        '''
        Loading the model to the device specified by the user.
        '''
        model_weights = self.model+'.bin'
        model_structure = self.model+'.xml'
        self.plugin = IECore()
#         logging.info('Plugin initialized.')        
            
        self.network = IENetwork(model_structure, model_weights)            
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
#         logging.info('IENetwork loaded into the plugin as self.exec_net.')
        
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))        
        
        return

    def predict(self, re_blob4infer, le_blob4infer, hp_blob4infer):
        '''
        Run predictions
        '''
        input_dir = {"right_eye_image": re_blob4infer, "left_eye_image":le_blob4infer, "head_pose_angles":hp_blob4infer}
        self.exec_net.start_async(request_id=0, inputs=input_dir)  
        if self.exec_net.requests[0].wait(-1) == 0:
            output = self.exec_net.requests[0].outputs["gaze_vector"]
        
        return output

    
#     def check_model(self):
#         raise NotImplementedError

        
    def preprocess_input(self, nparray, flag):
        '''
        Preprocess input frame.
        '''
        if flag=="hp":
            self.input_blob = "head_pose_angles" 
            blob4infer = nparray.reshape(1,3) #get model input shape
        else:
            if flag=="re":
                self.input_blob = "right_eye_image"
            elif flag=="le":
                self.input_blob = "left_eye_image"         
            model_shape = self.network.inputs[self.input_blob].shape 
            model_h, model_w = model_shape[2], model_shape[3]
            blob4infer = np.copy(nparray) #get ndarray
            blob4infer = cv2.resize(blob4infer, (model_w, model_h))
            blob4infer = blob4infer.transpose((2,0,1))
            blob4infer = blob4infer.reshape(1, 3, model_h, model_w)
            
        return blob4infer        


    def preprocess_output(self, gaze_vector, face_landmarks, frame_shape):
        '''
        Get required data from model output
        '''
        # get right eye and left eye coordinates
        x1, y1 = face_landmarks[0]
        x2, y2 = face_landmarks[1]
        # get point (0,0)
        x0 = (x2-x1)//2 + x1
        y0 = (y2-y1)//2 + y1

        h,w = frame_shape
        
        # Get mouse pointer position (x,y)
        # I quadrant  (in a Cartesian coordinate system in which z-axis is directed from person's eyes (mid-point between left and right eyes' centers) to the camera center, y-axis is vertical, and x-axis is orthogonal to both z,y axes)
        if gaze_vector[0][0] >= 0 and gaze_vector[0][1] >= 0:  
            x = x0 + int(gaze_vector[0][0]/(-gaze_vector[0][2])*(w - x0))
            y = y0 - int(gaze_vector[0][1]/(-gaze_vector[0][2])*y0)
        # II quadrant
        if gaze_vector[0][0] < 0 and gaze_vector[0][1] >= 0:  
            x = x0 + int(gaze_vector[0][0]/(-gaze_vector[0][2])*x0)
            y = y0 - int(gaze_vector[0][1]/(-gaze_vector[0][2])*y0)
        # III quadrant
        if gaze_vector[0][0] < 0 and gaze_vector[0][1] < 0:  
            x = x0 + int(gaze_vector[0][0]/(-gaze_vector[0][2])*x0)
            y = y0 - int(gaze_vector[0][1]/(-gaze_vector[0][2])*(h - y0))   
        # IV quadrant
        if gaze_vector[0][0] >= 0 and gaze_vector[0][1] <  0:  
            x = x0 + int(gaze_vector[0][0]/(-gaze_vector[0][2])*(w - x0))
            y = y0 - int(gaze_vector[0][1]/(-gaze_vector[0][2])*(h - y0))          
            
        mouse_pointer_position = (x, y)


        return mouse_pointer_position

