import common
import cv2
import sys
import time
import numpy as np
import face_recognition
import imutils
import pickle
import time
import cv2
from PIL import Image
sys.path.insert(0, '/home/pi/Desktop')
threshold = 0.2
top_k = 5  # number of objects to be shown as detected

model_dir = '/home/pi/Downloads'
model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
model_edgetpu = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
lbl = 'coco_labels.txt'

#with open('/home/pi/Downloads/coco_labels.txt', 'r') as f:
#    labels = [line.strip() for line in f.readlines()]

interpreter, labels = common.load_model(model_dir, model, lbl, edgetpu=0)
selected_labels = labels
counter = {label: 0 for label in labels}

currentname ="únknown"

encodingsP ="/home/pi/Desktop/lane_following/face/encodings.pickle"

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
 
cap = cv2.VideoCapture(0)


 
def getImg(display= False,size=[480,240]):
    _, img = cap.read()
    img = cv2.resize(img,(size[0],size[1]))
    if display:
        cv2.imshow('IMG',img)
    return img


def object_detection(display=False, size = [300, 300]):
    global interpreter, labels, threshold,currentname, encodingsP, data, cap
    _, frame = cap.read()
    img = cv2.resize(frame,(size[0],size[1]))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #============================================Object Detection
    pil_im = Image.fromarray(image)
    common.set_input(interpreter, pil_im)
    interpreter.invoke()
    objs = common.get_output(interpreter, score_threshold=0.2, top_k=5)
    frame = common.append_text_img1(frame, objs, labels,  counter, selected_labels)
    
    #====================================Face Recognition
    
    face_boxes = face_recognition.face_locations(img)
	# compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(img,face_boxes)
    names = []
	
    for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            face_counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                face_counts[name] = face_counts.get(name, 0) + 1
                
            name = max(face_counts, key=face_counts.get)

			#If someone in your dataset is identified, print their name on the screen
            if currentname != name:
                currentname = name
                print(currentname)

		# update the list of names
        names.append(name)

	# loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(face_boxes, names):
		# draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)

    if display:
        cv2.imshow('Object Detection',frame)
        
    return frame, currentname
    
 
if __name__ == '__main__':
    
    while True:
        img = getImg(True)
        
        frame, face_names = object_detection(True)
