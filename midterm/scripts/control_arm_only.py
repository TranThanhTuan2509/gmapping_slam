#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64

class ArmController:
    def __init__(self):
        rospy.init_node("arm_control", anonymous=True)
        
        # Publishers for servo control
        self.servo1_pub = rospy.Publisher("/meca/servo1_controller/command", Float64, queue_size=10)
        self.servo2_pub = rospy.Publisher("/meca/servo2_controller/command", Float64, queue_size=10)

    def run(self):
        """Continuously receives user input and sends servo commands."""
        while not rospy.is_shutdown():
            try:
                user_input = input("\nEnter 's1 s2' (servo1 servo2) or 'exit' to quit: ").strip()
                
                if user_input.lower() == "exit":
                    print("Exiting arm control.")
                    break

                s1, s2 = map(float, user_input.split())

                self.servo1_pub.publish(s1)
                self.servo2_pub.publish(s2)

                rospy.loginfo(f"Moving arms -> Servo1: {s1}, Servo2: {s2}")

            except ValueError:
                print("Invalid input. Please enter two numbers (s1 s2) or 'exit'.")

if __name__ == "__main__":
    arm_control = ArmController()
    arm_control.run()
