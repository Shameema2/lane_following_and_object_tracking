import common1 as cm
import cv2
import numpy as np
from PIL import Image
import time
from flask import Flask, Response, render_template
from threading import Thread
import speech_recognition as sr

app = Flask(__name__)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
threshold = 0.2
top_k = 10
edgetpu = 0

model_dir = '/home/pi/Downloads'
model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
model_edgetpu = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
lbl = 'coco_labels.txt'

counter = 0
prev_val = 0

r = sr.Recognizer()

def listen_for_command():
    with sr.Microphone() as source:
        print("Say something:")
        audio = r.listen(source)

    try:
        command = r.recognize_google(audio)
        print(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Error with the speech recognition service; {e}")
        return None

def show_selected_object_counter(objs, labels):
    global counter, prev_val, selected_obj
    arr = []
    for obj in objs:
        label = labels.get(obj.id, obj.id)
        arr.append(label)
            
    print("arr:", arr)
    global setected_obj
    x = arr.count(selected_obj)
    
    diff = x - prev_val
    
    print("diff:", diff)
    if diff > 0:
        counter = counter + diff
        
    prev_val = x
    
    print("counter:", counter)

def main():
    from util import edgetpu
    
    if edgetpu == 1:
        mdl = model_edgetpu
    else:
        mdl = model
    
    interpreter, labels = cm.load_model(model_dir, mdl, lbl, edgetpu)
    
    fps = 1
    arr_dur = [0, 0, 0]
    selected_obj = "person"
    
    while True:
        start_time = time.time()
        
        start_t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2_im = frame
        cv2_im = cv2.flip(cv2_im, 0)
        cv2_im = cv2.flip(cv2_im, 1)

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
       
        arr_dur[0] = time.time() - start_t0
        cm.time_elapsed(start_t0, "camera capture")
        
        start_t1 = time.time()
        cm.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = cm.get_output(interpreter, score_threshold=threshold, top_k=top_k)
        
        arr_dur[1] = time.time() - start_t1
        cm.time_elapsed(start_t1, "inference")
        
        start_t2 = time.time()
        show_selected_object_counter(objs, labels)
       
        arr_dur[2] = time.time() - start_t2
        cm.time_elapsed(start_t2, "other")
        cm.time_elapsed(start_time, "overall")
        
        cv2_im = cm.append_text_img1(cv2_im, objs, labels, arr_dur, counter, selected_obj)
        ret, jpeg = cv2.imencode('.jpg', cv2_im)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/video_feed')
def video_feed():
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)
    main()
