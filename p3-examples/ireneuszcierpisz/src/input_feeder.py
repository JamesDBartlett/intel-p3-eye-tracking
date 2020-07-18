'''
The class to feed input from an image, webcam, or video to a model.

'''
import cv2
from numpy import ndarray
import logging
logging.getLogger().setLevel(logging.INFO)

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for 'cam' input_type.
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
        if input_type == 'cam':
            self.input_file = 0
        self.cap = None
        self.wideo_writer = None
        
    
    def load_data(self):

        self.cap = cv2.VideoCapture(self.input_file)
        self.cap.open(self.input_file)
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) # gets Frame Rate
        print(f"fps:{fps:.2f}     frame stamp:{1/fps:.3f}sec")
        nf = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) # gets Number of Frames in the video file
        print(f"Number of frames in video: {nf}")
#         if self.input_type=='video':
#             self.cap=cv2.VideoCapture(self.input_file)
#             self.cap.open(self.input_file)

#         elif self.input_type=='cam':
#             self.cap=cv2.VideoCapture(0)
#             self.cap.open(self.input_file)
#         else:
#             self.cap=cv2.imread(self.input_file)

        if not self.cap.isOpened():
            logging.error("Unable to open video source!")
            
        # Grab the shape of the video frame 
        width = int(self.cap.get(3))
        height = int(self.cap.get(4))  
        vframe_shape = (height, width)
        
        if self.input_type == 'video':
            self.video_writer = cv2.VideoWriter('output_video.mp4', 0x00000021, 30, (width,height))        
            
        return vframe_shape
            
            
            
    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''          
            
#         while True:
#             for _ in range(5):
#                 _, frame=self.cap.read()
#             yield frame
#         while True:
#             _, frame=self.cap.read()
#             yield frame  
            
        while self.cap.isOpened():
            # Read the next frame
            flag, frame = self.cap.read()
            if not flag:
                break
            yield frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if self.input_type == 'video':
            self.video_writer.release()

        if not self.input_type=='image':
            self.cap.release()


    def write(self, frame_copy):
        self.video_writer.write(frame_copy)