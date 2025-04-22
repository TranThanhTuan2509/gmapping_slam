#!/usr/bin/env python3

import rospy
import sys
import tty
import termios
from geometry_msgs.msg import Twist
import time
import threading
import numpy as np
from std_msgs.msg import Float64

class TeleopController:
    def __init__(self):
        rospy.init_node("teleop_control", anonymous=True)
        self.pub = rospy.Publisher("/meca/diff_drive_controller/cmd_vel", Twist, queue_size=10)
        self.servo1_pub = rospy.Publisher("/meca/servo1_controller/command", Float64, queue_size=10)
        self.servo2_pub = rospy.Publisher("/meca/servo2_controller/command", Float64, queue_size=10)
        self.twist = Twist()
        self.velocity = 0.5  # Default linear velocity
        self.angular_velocity = 0.5  # Default angular velocity
        rospy.loginfo(
            "Press 'w' to move forward, "
            "'s' to move backward, "
            "'a' to rotate left, "
            "'d' to rotate right,\n"
            "'x' to stop.\n"
            "Press 'u' to increase speed, "
            "'l' to decrease speed. "
            "Current speed: {:.3f}".format(self.velocity)
        )

    def get_key(self):
        """Capture keyboard input"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def servo_oscillation(self):
        """Continuously moves the servo back and forth while the robot is running."""
        t = 0
        while not rospy.is_shutdown():
            servo1_value = 0.2 + 0.4 * np.sin(t)
            servo2_value = 0.5 + 0.4 * np.cos(t) 
            
            self.servo1_pub.publish(servo1_value)
            self.servo2_pub.publish(servo2_value)
            
            t += 0.1
            time.sleep(0.1)

    def run(self):
        servo_thread = threading.Thread(target=self.servo_oscillation)
        servo_thread.daemon = True
        servo_thread.start()
        while not rospy.is_shutdown():
            key = self.get_key()

            if key == "w":  # Move Forward
                self.twist.linear.x = self.velocity
                self.twist.linear.y = 0
                self.twist.angular.z = 0
            elif key == "s":  # Move Backward
                self.twist.linear.x = -self.velocity
                self.twist.linear.y = 0
                self.twist.angular.z = 0
            elif key == "a":  # Rotate Left
                self.twist.linear.x = 0
                self.twist.linear.y = 0
                self.twist.angular.z = self.angular_velocity
            elif key == "d":  # Rotate Right
                self.twist.linear.x = 0
                self.twist.linear.y = 0
                self.twist.angular.z = -self.angular_velocity
            elif key == "l":  # Decrease Speed
                self.velocity = max(0.1, self.velocity - 0.001)
                self.angular_velocity = max(0.1, self.angular_velocity - 0.001)
                rospy.loginfo(f"Speed Decreased: {self.velocity}")
            elif key == "u":  # Increase Speed
                self.velocity += 0.1
                self.angular_velocity += 0.1
                rospy.loginfo(f"Speed Increased: {self.velocity}")
            elif key == "x":  # Stop
                self.twist.linear.x = 0
                self.twist.linear.y = 0
                self.twist.angular.z = 0
                rospy.loginfo("Stopping")
            elif key == "\x03":  # Ctrl+C to exit
                break

            self.pub.publish(self.twist)

if __name__ == "__main__":
    teleop = TeleopController()
    teleop.run()
