import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util
import pyttsx3
from threading import Thread
import common1 as cm
from PIL import Image



sys.path.insert(0, '/home/pi/Desktop')
import util as ut
ut.init_gpio()

from flask import Flask, Response
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    #return "Default Message"
    return render_template("index2.html")

@app.route('/video_feed')
def video_feed():
    #global cap
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def mySpeak(message):
    engine = pyttsx3.init()
    engine.say('{}'.format(message))
    engine.runAndWait()
    
    

mySpeak('''Hi shameema''')
mySpeak('''Turn on object detection''')

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0, cv2.CAP_V4L2)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        
        
        
def main():
    from util import edgetpu
    if (edgetpu==1):
        mdl = model_edgetpu
    else:
        mdl=model
        
    interpreter, labels = cm.load_model(model_dir, mdl, lbl, edgetpu)
    
    fps=1
    arr_dur=[0,0,0]
    
    while True:
        start_time = time.time()
        start_t0 = time.time()
        
        t1=  cv2.getTickCount()
        
        ret, frame1 = videoStream.read()
        
threshold=0.2
top_k=5 #number of objects to be shown as detected

model_dir = '/home/pi/Downloads'
model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
model_edgetpu = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
lbl = 'coco_labels.txt'

counter=0
prev_val=0

file_path='/home/pi/Desktop/ob.code/web/'
selected_obj="person"
prev_val_obj=""
        
def main(modeldir=model_dir, labels=lbl, threshold=0.2, edgetpu = model_edgetpu, graph= model):
    # Convert resolution argument to tuple
    use_TPU= edgetpu
    #resW, resH = map(int, resolution.split('x'))
    imW, imH = 640, 480

    # Convert edgetpu argument to boolean
    use_TPU = edgetpu

    # Import TensorFlow libraries
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        if graph == model:
            graph = edgetpu

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(modeldir, graph)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(modeldir, labels)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize video stream
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)

    # Create window
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

    while True:
        t1 = cv2.getTickCount()
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i in range(len(scores)):
            if 0.5 <= scores[i] <= 1.0:
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                message = label
                mySpeak(message)

        cv2.imshow('Object detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2204, threaded = True)
    main(modeldir=model_dir, labels=lbl, threshold=0.2, edgetpu = model_edgetpu, graph= model)