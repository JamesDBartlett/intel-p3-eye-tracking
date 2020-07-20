'''
 The Class for a model. 
'''

from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import logging
logging.getLogger().setLevel(logging.INFO)

class Model_Landmarks:
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

    def predict(self, image):
        '''
        Run predictions on the input image.
        '''
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: image})  
        if self.exec_net.requests[0].wait(-1) == 0:
            output = self.exec_net.requests[0].outputs[self.output_blob]   

        return output

    
#     def check_model(self):
#         raise NotImplementedError

        
    def preprocess_input(self, frame):
        '''
        Preprocess input frame.
        '''
        model_shape = self.network.inputs[self.input_blob].shape 
        model_w = model_shape[3]
        model_h = model_shape[2]

        # copying the frame as numpy.ndarray and assignes the returning copy to the frame4infer variable.
        frame4infer = np.copy(frame)
        frame4infer = cv2.resize(frame4infer, (model_w, model_h))
        frame4infer = frame4infer.transpose((2,0,1))
        frame4infer = frame4infer.reshape(1, 3, model_h, model_w)
    
        return frame4infer        


    def preprocess_output(self, output, face_bb_coord):
        '''
        Get required data from model output
        '''
#        print('landmarks output', output[0], output.shape, len(output[0]))
#         print('face_bb_coord', face_bb_coord)
        x1,y1,x2,y2 = face_bb_coord
        w,h = x2-x1, y2-y1

        #output is a row-vector of 10 floating point values for five landmarks coordinates
        #two eyes, nose, and two lip corners. # The output shape is 1X10 
        # ['right_eye', 'left_eye', 'nose', 'right_lip_corner', 'left_lip_corner']
        # get list of values of the five landmarks as tuples:
        l = []
        for i in range(0,len(output[0]),2):
            l.append((output[0][i][0][0], output[0][i+1][0][0]))
#         print(l)
        # get coordinates of the five landmarks
        right_eye = (int(l[0][0]*w+x1), int(l[0][1]*h+y1))
        left_eye = (int(l[1][0]*w+x1), int(l[1][1]*h+y1))
        nose = (int(l[2][0]*w+x1), int(l[2][1]*h+y1))
        right_lip_corner = (int(l[3][0]*w+x1), int(l[3][1]*h+y1))
        left_lip_corner = (int(l[4][0]*w+x1), int(l[4][1]*h+y1))
        
#         print('right_eye, left_eye', right_eye, left_eye)
        landmarks_coords = (right_eye, left_eye, nose, right_lip_corner, left_lip_corner)

        return landmarks_coords

