import numpy as np
import cv2
import sys
import utils
import logging as log
from openvino.inference_engine import IECore

class Model:
    """
    Generic abstract class to do inference on different networks
    """
    IE=None
    net=None
    exec_net=None
    device=None

    def __init__(self, model_xml):
        self.IE=IECore()
        self.net=self.IE.read_network(model=model_xml,weights=model_xml.replace('xml','bin'))

    def __check_layers__(self):
        layers_map = self.IE.query_network(network=self.net,device_name=self.device)
        for layer in self.net.layers.keys():
            if layers_map.get(layer, "none") == "none": #Found unsupported layer
                return False
        return True

    def load_model(self, device_name='CPU'):
        self.device=device_name
        if(self.__check_layers__()):
            self.exec_net=self.IE.load_network(network=self.net,device_name=device_name,num_requests=1)
        else:
            log.critical("Unsupported layer found, can't continue")
            sys.exit(1)

    def predict(self, image, req_id):
        input_name = next(iter(self.net.inputs))
        input_dict={input_name:image}
        request_handle=self.exec_net.start_async(request_id=req_id, inputs=input_dict)
        return request_handle

    def get_output(self, request_handle):
        pass

    def preprocess_input(self, image):
        pass