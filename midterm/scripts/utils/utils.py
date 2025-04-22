#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
import numpy as np
import threading
import time

class Controller:
    def __init__(self):
        rospy.init_node("mecanum_velocity_controller", anonymous=True)
        
        # Publishers for wheel velocities
        self.pub_lf = rospy.Publisher("/meca/left_forward_controller/command", Float64, queue_size=10)
        self.pub_lb = rospy.Publisher("/meca/left_backward_controller/command", Float64, queue_size=10)
        self.pub_rf = rospy.Publisher("/meca/right_forward_controller/command", Float64, queue_size=10)
        self.pub_rb = rospy.Publisher("/meca/right_backward_controller/command", Float64, queue_size=10)
        
        # Publishers for servo motors
        self.servo1_pub = rospy.Publisher("/meca/servo1_controller/command", Float64, queue_size=10)
        self.servo2_pub = rospy.Publisher("/meca/servo2_controller/command", Float64, queue_size=10)

        # LiDAR Subscriber
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.scan_data = None  # Store LiDAR scan data
        
        self.rate = rospy.Rate(10)  # 10 Hz update rate

        # Robot parameters
        self.lx = 0.148  # Distance from center to wheels in x direction (m)
        self.ly = 0.075  # Distance from center to wheels in y direction (m)
        self.r = 0.05983 # Wheel radius (m)

        # Thresholds for obstacle detection
        self.OBSTACLE_DISTANCE_THRESHOLD = 0.5  # Meters
        self.LEFT_REGION = range(90, 150)  # Angles covering left side
        self.RIGHT_REGION = range(210, 270)  # Angles covering right side

    def lidar_callback(self, msg):
        """Callback function for LiDAR data."""
        self.scan_data = msg.ranges

    def get_obstacle_direction(self):
        """Determines whether an obstacle is on the left or right side."""
        if self.scan_data is None:
            return None  # No data
        
        left_distances = [self.scan_data[i] for i in self.LEFT_REGION if not np.isnan(self.scan_data[i])]
        right_distances = [self.scan_data[i] for i in self.RIGHT_REGION if not np.isnan(self.scan_data[i])]
        
        min_left = min(left_distances) if left_distances else float("inf")
        min_right = min(right_distances) if right_distances else float("inf")

        if min_left < self.OBSTACLE_DISTANCE_THRESHOLD and min_right < self.OBSTACLE_DISTANCE_THRESHOLD:
            return 'both'
        elif min_left < self.OBSTACLE_DISTANCE_THRESHOLD:
            return "left"
        elif min_right < self.OBSTACLE_DISTANCE_THRESHOLD:
            return "right"
        return "clear"
    
    def servo_oscillation(self):
        """Continuously moves the servo back and forth while the robot is running."""
        t = 0
        while not rospy.is_shutdown():
            servo1_value = 0.5 + 0.4 * np.sin(t)  # Oscillates between 0.1 and 0.9
            servo2_value = 0.1 + 0.4 * np.cos(t)  # Oscillates between 0.1 and 0.9
            
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
    
    def stop(self):
            """Stops the robot by setting all velocities to zero."""
            rospy.loginfo("Stopping robot...")
            self.pub_lf.publish(0.0)
            self.pub_rf.publish(0.0)
            self.pub_lb.publish(0.0)
            self.pub_rb.publish(0.0)
            self.servo1_pub.publish(0.0)
            self.servo2_pub.publish(0.0)

    def move(self):
        """Moves the robot with obstacle avoidance."""
        rospy.loginfo("Initializing movement...")
        rospy.sleep(2)  # Allow ROS to initialize
        rospy.loginfo("Running...")
        
        servo_thread = threading.Thread(target=self.servo_oscillation)
        servo_thread.daemon = True
        servo_thread.start()

        while not rospy.is_shutdown():
            try:
                user_input = input("\nEnter 'vx vy wz' (velocities) or 'stop' to stop: ").strip()
                
                if user_input.lower() == "stop":
                    self.pub_lf.publish(0.0)
                    self.pub_lb.publish(0.0)
                    self.pub_rf.publish(0.0)
                    self.pub_rb.publish(0.0)
                    print("Robot Stopped!")
                    continue  # Skip the rest of the loop

                # Parse user input
                vx, vy, omega = map(float, user_input.split())

                # Detect obstacles
                obstacle_direction = self.get_obstacle_direction()

                if obstacle_direction == "both":
                    rospy.logwarn("Obstacle detected on both directions! STOP...")
                    vx, vy, omega = 0.0, 0.0, 0.0  # Stop the robot
                elif obstacle_direction == "left":
                    rospy.logwarn("Obstacle detected on LEFT! Turning RIGHT...")
                    vx, vy, omega = 0.0, 0.0, -1.0  # Turn right
                elif obstacle_direction == "right":
                    rospy.logwarn("Obstacle detected on RIGHT! Turning LEFT...")
                    vx, vy, omega = 0.0, 0.0, 1.0  # Turn left
                else:
                    rospy.loginfo("No obstacles detected. Moving forward.")

                # Compute wheel velocities using inverse kinematics
                wheel_velocities = self.inverse_kinematics(vx, vy, omega)

                # Publish wheel velocities
                self.pub_lf.publish(wheel_velocities[0])
                self.pub_rf.publish(wheel_velocities[1])
                self.pub_lb.publish(wheel_velocities[2])
                self.pub_rb.publish(wheel_velocities[3])

                # Correct logging
                rospy.loginfo(f"Moving Wheels -> LF: {wheel_velocities[0]:.2f}, "
                            f"LB: {wheel_velocities[2]:.2f}, RF: {wheel_velocities[1]:.2f}, "
                            f"RB: {wheel_velocities[3]:.2f}")

            except ValueError:
                print("Invalid input. Please enter 3 numbers (vx vy wz) or 'stop'.")

        # rospy.sleep(5)  # Allow ROS to initialize
        # Stop the robot
        self.stop()

if __name__ == "__main__":
    controller = Controller()