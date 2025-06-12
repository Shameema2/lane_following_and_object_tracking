

# ========== Imports Start
import cv2

from LaneDetectionModule import getLaneCurve
import WebcamModule
import utlis

import gpiozero_robot





# ========== Main Function Start
def main():
    img = WebcamModule.getImg(display=False)
    frame, face_names = WebcamModule.object_detection(display=True)
    
    curveVal = getLaneCurve(img, 0)

    sen = 1.3  
    
    # lithium battery can deliver high current, so use value lower than 1,
    # adapter delivers very low current(180mA), so use 1(full available speed)
    maxVal = 1  # MAX SPEED
    if curveVal > maxVal:
        curveVal = maxVal
    if curveVal < -maxVal:
        curveVal = -maxVal
    print("curve value = " + str(curveVal))
    if curveVal > 0:      # Deadzone, if in this -0.08 to 0.05 then no turning
        sen = 1.7
        if curveVal < 0.05:
            curveVal = 0
    else:
        if curveVal > -0.08:
            curveVal = 0
            
    
    
    if 'senjuty' in face_names:
        gpiozero_robot.move(curveVal)
        

# ========== Main Function End


# ========== If this module is run
if __name__ == '__main__':
    intialTrackBarVals = [85, 15, 30, 240]
    utlis.initializeTrackbars(intialTrackBarVals)
    while True:
        main()
            
            
        if cv2.waitKey(1) & 0xff == ord('q'):
            gpiozero_robot.stopping()
            cv2.destroyAllWindows()
            break
    
