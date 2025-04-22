#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import torch
import numpy as np
import os
import sys
import math
import tf.transformations

# Add yolov5 repo to path
FILE = os.path.dirname(os.path.realpath(__file__))
YOLO_PATH = os.path.join(FILE, "yolov5")
if YOLO_PATH not in sys.path:
    sys.path.append(YOLO_PATH)

class HumanFollower:
    def __init__(self):
        # Initialize ROS node and components
        rospy.init_node('human_detection', anonymous=True)
        self.bridge = CvBridge()
        self.cmd_vel_pub = rospy.Publisher("/meca/diff_drive_controller/cmd_vel", Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber("/midterm/odom", Odometry, self.odom_callback)

        # Load YOLOv5 model
        rospy.loginfo("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = 0.4  # Confidence threshold
        rospy.loginfo("YOLOv5 model loaded.")

        # Subscribe to camera topic
        self.image_sub = rospy.Subscriber("/midterm/camera/rgb/image_raw", Image, self.image_callback)

        # State variables
        self.human_detected = False
        self.searching = True
        self.reached = False
        self.rate = rospy.Rate(10)  # 10 Hz
        self.start_time = None      # To track rotation start time
        self.total_rotation = 0.0   # Accumulated rotation in radians
        self.rotation_speed = 0.5   # Rotation speed in rad/s
        self.max_rotation = 2 * math.pi  # 360 degrees in radians
        self.count = 0
        self.max_num = 13

        # Odometry variables
        self.current_position = None
        self.current_orientation = None

        rospy.loginfo("YOLOv5 Human detection node started.")

    def odom_callback(self, data):
        # Update current position and orientation
        self.current_position = data.pose.pose.position
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        self.current_orientation = yaw

    def move_linear(self, distance):
        """Move the robot a specified distance using odometry."""
        if self.current_position is None:
            rospy.logwarn("Odometry data not available. Cannot move linearly.")
            return

        target_distance = abs(distance)
        direction = -1 if distance < 0 else 1  # Negative for backward movement
        start_position = self.current_position
        twist = Twist()
        twist.linear.x = direction * 0.2  # Speed: 0.2 m/s

        while not rospy.is_shutdown():
            dx = self.current_position.x - start_position.x
            dy = self.current_position.y - start_position.y
            distance_traveled = math.sqrt(dx**2 + dy**2)

            if distance_traveled >= target_distance:
                break

            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()

        self.cmd_vel_pub.publish(Twist())  # Stop the robot
        rospy.loginfo("Linear movement completed. Distance traveled: %.2f meters", distance_traveled)

    def rotate_angular(self, angle):
        """Rotate the robot by a specified angle using odometry."""
        if self.current_orientation is None:
            rospy.logwarn("Odometry data not available. Cannot rotate.")
            return

        target_orientation = (self.current_orientation + angle) % (2 * math.pi)
        twist = Twist()
        twist.angular.z = 0.5  # Rotation speed: 0.5 rad/s

        while not rospy.is_shutdown():
            diff = (target_orientation - self.current_orientation) % (2 * math.pi)
            if diff > math.pi:
                diff -= 2 * math.pi

            if abs(diff) < 0.05:  # Tolerance of ~3 degrees
                break

            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()

        self.cmd_vel_pub.publish(Twist())  # Stop the robot
        rospy.loginfo("Rotation completed. Angle rotated: %.2f radians", angle)

    def image_callback(self, data):
        # Skip processing if the task is complete
        if self.reached:
            return

        try:
            # Convert ROS image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            frame = cv2.resize(frame, (640, 480))

            # Perform YOLOv5 detection
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()
            human_boxes = [d for d in detections if int(d[5]) == 0]  # Class 0 is 'person'

            if len(human_boxes) > 0:
                # Human detected
                self.human_detected = True
                self.searching = False

                # Select the largest (closest) human
                largest = max(human_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x1, y1, x2, y2, conf, cls = largest
                h = int(y2 - y1)
                # distance_estimate = 5000.0 / h  # Simple distance estimation
                # rospy.loginfo("Human detected. Estimated distance: %.2f meters", distance_estimate)
                #
                # # Move backward using odometry
                # self.move_linear(-distance_estimate)
                #
                # # Rotate 180 degrees using odometry
                # rospy.loginfo("Rotating 180 degrees...")
                # self.rotate_angular(math.pi)  # 180 degrees in radians
                #
                # rospy.loginfo("Reached human. Task complete.")
                # self.reached = True

            # else:
            #     # No human detected
            #     if self.searching:
            #         # Start tracking rotation if not already started
            #         if self.start_time is None:
            #             self.start_time = time.time()
            #             rospy.loginfo("No human detected. Starting 360-degree search...")
            #         self.count += 1
            #         # Calculate total rotation
            #         elapsed_time = time.time() - self.start_time
            #         self.total_rotation = self.rotation_speed * elapsed_time

                    # if self.count >= self.max_num:
                    #     # Completed 360 degrees without detection
                    #     rospy.loginfo("No human detected after 360-degree rotation. Stopping.")
                    #     self.cmd_vel_pub.publish(Twist())  # Stop the robot
                    #     self.reached = True
                    # else:
                    #     # Continue searching by rotating
                    #     twist = Twist()
                    #     twist.angular.z = self.rotation_speed
                    #     self.cmd_vel_pub.publish(twist)
                    #     time.sleep(5)
                    #     rospy.loginfo_throttle(5, "Searching for human... Rotating.")

            # Visualization
            for b in human_boxes:
                x1, y1, x2, y2 = map(int, b[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {b[4]:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Rear Camera - YOLOv5 Human Detection", frame)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr("Error in image_callback: %s", str(e))

if __name__ == '__main__':
    try:
        follower = HumanFollower()
        rospy.spin()  # Keep the node running until interrupted
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
