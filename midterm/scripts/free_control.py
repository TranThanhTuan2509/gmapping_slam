#!/usr/bin/env python3

import rospy
import sys
import time
import threading
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import tf.transformations

class TeleopController:
    def __init__(self):
        rospy.init_node("free_control", anonymous=True)
        self.pub = rospy.Publisher("/meca/diff_drive_controller/cmd_vel", Twist, queue_size=10)
        self.servo1_pub = rospy.Publisher("/meca/servo1_controller/command", Float64, queue_size=10)
        self.servo2_pub = rospy.Publisher("/meca/servo2_controller/command", Float64, queue_size=10)
        
        self.odom_sub = rospy.Subscriber('/midterm/odom', Odometry, self.odom_callback)

        self.twist = Twist()
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0 

    def servo_oscillation(self):
        """Continuously moves the servos back and forth while the robot is running."""
        t = 0
        while not rospy.is_shutdown():
            servo1_value = 0.2 + 0.4 * np.sin(t) 
            servo2_value = 0.5 + 0.4 * np.cos(t) 
            
            self.servo1_pub.publish(servo1_value)
            self.servo2_pub.publish(servo2_value)
            
            t += 0.1
            time.sleep(0.1)

    def odom_callback(self, msg):
        """Callback function to update robot's position and yaw angle."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion(quaternion)

    def has_reached_goal(self, target_x=None, target_yaw=None, tolerance=0.2):
        """Check if the robot has reached the goal position or rotation."""
        if target_x is not None:
            distance = abs(target_x - self.current_x)
            if distance < tolerance:
                return True

        if target_yaw is not None:
            angle_diff = abs(target_yaw - self.current_yaw)
            if angle_diff < tolerance:
                return True

        return False
    
    def stop(self):
        """Stops the robot by publishing zero velocities."""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.pub.publish(self.twist)
        rospy.loginfo("Robot stopped.")

    def run(self):
        """Moves the robot towards a goal while avoiding obstacles."""
        servo_thread = threading.Thread(target=self.servo_oscillation)
        servo_thread.daemon = True
        servo_thread.start()
        
        while not rospy.is_shutdown():
            try:
                user_input = input("\nEnter 'x z' (linear_x angular_z) or 'stop' to stop: ").strip()
                
                if user_input.lower() == "stop":
                    self.stop()
                    print("Robot Stopped!")
                    break

                velocity_x, velocity_z = map(float, user_input.split())

                if velocity_z != 0:  
                    target_yaw = self.current_yaw + velocity_z 

                    while not self.has_reached_goal(target_yaw=target_yaw, tolerance=0.09): 
                        self.twist.angular.z = velocity_z
                        self.twist.linear.x = 0.0
                        self.pub.publish(self.twist)

                        rospy.loginfo(f"Rotating -> Target: {target_yaw:.2f} rad, Current: {self.current_yaw:.2f} rad")
                        rospy.sleep(0.1) 

                    self.stop()
                    rospy.loginfo("Rotation goal reached! Robot stopped.")
                    break 

                if velocity_x != 0:  
                    target_x = self.current_x + velocity_x  
                    while not self.has_reached_goal(target_x=target_x):
                        self.twist.linear.x = velocity_x
                        self.twist.angular.z = 0.0
                        self.pub.publish(self.twist)

                        rospy.loginfo(f"Moving -> Target X: {target_x:.2f}, Current X: {self.current_x:.2f}")
                        rospy.sleep(0.1) 

                    self.stop()  
                    rospy.loginfo("Position goal reached! Robot stopped.")
                    break

            except ValueError:
                print("Invalid input. Please enter two numbers (x z) or 'stop'.")

if __name__ == "__main__":
    teleop = TeleopController()
    teleop.run()
