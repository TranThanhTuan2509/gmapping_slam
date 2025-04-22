#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
import numpy as np
import sys
import termios
import tty
import time
import threading

class CustomTeleopController:
    def __init__(self):
        rospy.init_node("custom_teleop_controller", anonymous=True)


        self.pub_lf = rospy.Publisher("/meca/left_forward_controller/command", Float64, queue_size=10)
        self.pub_lb = rospy.Publisher("/meca/left_backward_controller/command", Float64, queue_size=10)
        self.pub_rf = rospy.Publisher("/meca/right_forward_controller/command", Float64, queue_size=10)
        self.pub_rb = rospy.Publisher("/meca/right_backward_controller/command", Float64, queue_size=10)

        self.servo1_pub = rospy.Publisher("/meca/servo1_controller/command", Float64, queue_size=10)
        self.servo2_pub = rospy.Publisher("/meca/servo2_controller/command", Float64, queue_size=10)
        
        self.rate = rospy.Rate(10)  # 10 Hz update rate
   
        self.lx = 0.148  # Distance from center to wheels in x direction (m)
        self.ly = 0.075  # Distance from center to wheels in y direction (m)
        self.r = 0.05983 # Wheel radius (m)

  
        self.velocity = 0.2  # Initial velocity
        self.velocity_step = 0.001  # Amount to increase/decrease per key press
        self.max_velocity = 1.0  # Maximum velocity
        self.min_velocity = -1.0  # Minimum velocity

        rospy.loginfo("Custom Teleop Control Ready! Use keys to control the robot.")

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

    def inverse_kinematics(self, vx, vy, omega=0):
        """Computes wheel velocities from desired base velocity."""
        L = self.lx + self.ly  
        J_inv = np.array([
            [1, -1, -L],
            [1,  1,  L],
            [1,  1, -L],
            [1, -1,  L]
        ]) / self.r
        return np.dot(J_inv, np.array([vx, vy, omega]))

    def publish_wheel_speeds(self, vx, vy, omega):
        """Publishes the calculated wheel speeds."""
        wheel_velocities = self.inverse_kinematics(vx, vy, omega)
        self.pub_lf.publish(wheel_velocities[0])
        self.pub_rf.publish(wheel_velocities[1])
        self.pub_lb.publish(wheel_velocities[2])
        self.pub_rb.publish(wheel_velocities[3])

    def get_key(self):
        """Captures a single key press."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def run(self):
        """Runs the teleop control loop."""
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

        servo_thread = threading.Thread(target=self.servo_oscillation)
        servo_thread.daemon = True
        servo_thread.start()
        while not rospy.is_shutdown():
            rospy.loginfo(f"CURRENT VELOCITY {self.velocity}")
            key = self.get_key()

            if key == "u":  # Increase speed
                self.velocity = min(self.velocity + self.velocity_step, self.max_velocity)
                rospy.loginfo("Increased speed: {:.3f}".format(self.velocity))
            
            elif key == "l":  # Decrease speed
                self.velocity = max(self.velocity - self.velocity_step, self.min_velocity)
                rospy.loginfo("Decreased speed: {:.3f}".format(self.velocity))

            elif key == "w":  # Forward
                self.publish_wheel_speeds(self.velocity, 0.0, 0.0)
            
            elif key == "s":  # Backward
                self.publish_wheel_speeds(-self.velocity, 0.0, 0.0)
            
            elif key == "a":  # Rotate Left
                self.publish_wheel_speeds(0.0, 0.0, self.velocity)
            
            elif key == "d":  # Rotate Right
                self.publish_wheel_speeds(0.0, 0.0, -self.velocity)
            
            elif key == "x":  # Stop
                self.publish_wheel_speeds(0.0, 0.0, 0.0)
                rospy.loginfo("Robot stopped.")

            elif key == "\x03":  # Ctrl+C to exit
                break

if __name__ == "__main__":
    controller = CustomTeleopController()
    controller.run()
