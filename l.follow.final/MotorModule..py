
import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
 
 
class Motor():
    def __init__(self,EN1,In1A,In2A,EN2,In1B,In2B):
        self.EN1 = EN1
        self.In1A = In1A
        self.In2A = In2A
        self.EN2 = EN2
        self.In1B = In1B
        self.In2B = In2B
        GPIO.setup(self.EN1,GPIO.OUT);
        GPIO.setup(self.In1A,GPIO.OUT);
        GPIO.setup(self.In2A,GPIO.OUT);
        GPIO.setup(self.EN2,GPIO.OUT);
        GPIO.setup(self.In1B,GPIO.OUT);
        GPIO.setup(self.In2B,GPIO.OUT);
        self.pwm1 = GPIO.PWM(self.EN1, 200);
        self.pwm2 = GPIO.PWM(self.EN2, 200);
        self.pwm1.start(0);
        self.pwm2.start(0);
        self.mySpeed=0
 
    def move(self, speed=0.5, turn=0, t=0):
        speed *= 100
        turn *= 70        # todo: starnge value, need to check video
        leftSpeed = speed-turn
        rightSpeed = speed+turn
 
        if leftSpeed>100: leftSpeed =100
        elif leftSpeed<-100: leftSpeed = -100
        if rightSpeed>100: rightSpeed =100
        elif rightSpeed<-100: rightSpeed = -100
        # print(leftSpeed,rightSpeed)
        self.pwm2.ChangeDutyCycle(abs(leftSpeed))
        self.pwm1.ChangeDutyCycle(abs(rightSpeed))
        
        if (leftSpeed>0):
            GPIO.output(self.In1A,GPIO.HIGH);GPIO.output(self.In2A,GPIO.LOW)
        else:
            GPIO.output(self.In1A,GPIO.LOW);GPIO.output(self.In2A,GPIO.HIGH)
        if (rightSpeed>0):
            GPIO.output(self.In1B,GPIO.HIGH);GPIO.output(self.In2B,GPIO.LOW)
        else:
            GPIO.output(self.In1B,GPIO.LOW);GPIO.output(self.In2B,GPIO.HIGH)
        sleep(t)
 
    def stop(self,t=0):
        self.pwm1.ChangeDutyCycle(0);
        self.pwm2.ChangeDutyCycle(0);
        self.mySpeed=0
        sleep(t)
 
def main():
    motor.move(0.5, 0, 2)
    motor.stop(2)
    motor.move(-0.5, 0, 2)
    motor.stop(2)
    motor.move(0, 0.5, 2)
    motor.stop(2)
    motor.move(0, -0.5, 2)
    motor.stop(2)
 
if __name__ == '__main__':
    try:
        motor= Motor(23, 16, 12,24,  21, 18)
        main()
    except KeyboardInterrupt:
        print("terminated")
        
    finally:
        GPIO.cleanup()
    
