"""
Motor pins are hardcoded in this module.
The car will take very small turn and correct direction. Then go forward.
Test to find out usable minimum values for speed and duration.
Usable function: 
    move()
    stopping()
"""

from gpiozero import Robot, PWMOutputDevice
import time

# ========== Minimum speed and duration values start
# todo: Need to find out by testing. Use minimum usable value
min_turn_speed = 0.8
turn_duration = 0.1

min_forward_speed = 0.9
forward_duration = 0.6
# ========== Minimum speed and duration values end

# ========== Motor pin definition
robot = Robot(right=(16, 12), left=(21, 18))
right_motor_en = PWMOutputDevice(23,frequency=2000)
left_motor_en = PWMOutputDevice(24,frequency=2000)

def move(curve):
    """
    Runs the motors to go forward or turn left/right 
    @ min speed and for min duration.
    Then stops the car.
    
    input: curve value
    output: prints direction of movement
    """
    right_motor_en.value = min_turn_speed
    
    left_motor_en.value= min_turn_speed
    if curve > 0.0:
        robot.right(min_turn_speed)
        print("           right\n")
        time.sleep(turn_duration)

    elif curve < 0.0:
        robot.left(min_turn_speed)
        print("left\n")
        time.sleep(turn_duration)
        

    elif curve == 0:
        robot.forward(min_forward_speed)
        print("          forward          \n")
        time.sleep(forward_duration)

    robot.stop()


def stopping():
    """stops the car"""
    robot.stop()
    right_motor_en.value=0
    left_motor_en.value=0
