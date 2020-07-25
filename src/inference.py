#!/usr/bin/env python3


"""
NOTE TO UDACITY MENTORS:
This file is based primarily on the inference.py found in the official Intel IoT Devkit "People Counter Python" repo on GitHub:
https://github.com/intel-iot-devkit/people-counter-python/blob/master/inference.py

Very little was needed in the way of changes to this file, so it bears close resemblance to its original source,
other than some basic formatting differences. 
"""


"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(
        self,
        model,
        device = "CPU",
        cpu_extension=None,
    ):
        """
         Loads a network and an image to the Inference Engine plugin.
        :param model: .xml file of pre trained model
        :param cpu_extension: extension for the CPU device
        :param device: Target device
        :param input_size: Number of input layers
        :param output_size: Number of output layers
        :param num_requests: Index of Infer request value. Limited to device capabilities.
        :param plugin: Plugin for specified device
        :return:  Shape of input layer
        """

        model_bin = os.path.splitext(model)[0] + ".bin"
        self.plugin = IECore()

        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read IR
        log.info("Reading IR...")
        self.net = self.plugin.read_network(model=model, weights=model_bin)

        log.info("Loading IR to the plugin...")
        supported_layers = self.plugin.query_network(network=self.net, device_name=device)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        self.exec_network = self.plugin.load_network(self.net, device)
        
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))

    def get_input_shape(self):
        """
        Gives the shape of the input layer of the network.
        :return: None
        """
        return self.net.inputs[self.input_blob].shape

    def performance_counter(self, request_id):
        """
        Queries performance measures per layer to get feedback of what is the
        most time consuming layer.
        :param request_id: Index of Infer request value. Limited to device capabilities
        :return: Performance of the layer  
        """
        perf_count = self.net_plugin.requests[request_id].get_perf_counts()
        return perf_count

    def exec_network(self, input1, input2=None, input3=None):
        """
        Starts asynchronous inference for specified request.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param frame: Input image
        :return: Instance of Executable Network class
        """
        inputiter = iter(self.net.inputs)

        if(input2 is None and input3 is None):
            self.exec_network.start_async(request_id=0, 
                inputs={self.input_blob: input1})
        elif(input3 is None):    
            next(inputiter)#ignore first input
            self.exec_network.start_async(request_id=0, 
                inputs={self.input_blob: input1, next(inputiter):input2})
        else:
            next(inputiter)#ignore first input
            self.exec_network.start_async(request_id=0, 
                inputs={self.input_blob: input1, next(inputiter):input2, next(inputiter):input3})

        return
        

    def wait(self):
        """
        Waits for the result to become available.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :return: Timeout value
        """
        return self.exec_network.requests[0].wait(-1)

    def get_output(self):
        """
        Gives a list of results for the output layer of the network.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param output: Name of the output layer
        :return: Results for the specified request
        """
        return self.exec_network.requests[0].outputs

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.net_plugin
        del self.plugin
        del self.net
