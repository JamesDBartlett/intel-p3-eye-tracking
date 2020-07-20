'''
 The Class for a model. 
'''

from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import logging
logging.getLogger().setLevel(logging.INFO)

class Model_HeadPose:
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

    
    def list_output_keys(self):
        """
        Gets the network output layer blobs list
        """
        key_list = list(self.network.outputs.keys())
        
        return key_list
    
    
    def predict(self, image):
        '''
        Run predictions on the input image.
        '''
        # headpose output blobs: 'angle_y_fc','angle_p_fc', 'angle_r_fc'
        
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: image})  
        if self.exec_net.requests[0].wait(-1) == 0:
            yaw = self.exec_net.requests[0].outputs['angle_y_fc']            
            pitch = self.exec_net.requests[0].outputs['angle_p_fc']   
            roll = self.exec_net.requests[0].outputs['angle_r_fc']
   
        return yaw, pitch, roll

    
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


    def preprocess_output(self, outputs):
        '''
        Get required data from model output
        '''
        yaw, pitch, roll = outputs
#         print('yaw, pitch, roll:', yaw, pitch, roll)
        headpose_angles = [yaw[0][0], pitch[0][0], roll[0][0]]

        return headpose_angles

